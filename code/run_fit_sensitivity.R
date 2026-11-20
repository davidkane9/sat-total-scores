# Sensitivity refits for the manuscript's Sensitivity section.
#
# Fits three variants of the canonical SHASH+Dirichlet model to the SAME
# imputed dataset (imputation m = 1 of the canonical MI run, reproduced
# here by seeding exactly as collect_one(1) does in run_fit_mi.R), so
# that the comparison isolates the modelling choice from the rounding
# imputation.  The canonical fit for that imputation already exists as
# ../replicated/sims/mi/imp_01.rds and is NOT refit; it is the reference the
# variants are compared against.
#
#   skewt     Fernandez-Steel two-piece skew-t in place of sinh-arcsinh
#             (skewt_dirichlet_bands.stan).  Alternative parametric tail
#             family: per-group skewness gamma_r and degrees of freedom
#             nu_r replace epsilon_r and delta_r.
#   widedelta Canonical model with the between-group tail-weight
#             hyperprior widened five-fold, tau_log_delta ~ N+(0, 1.0)
#             instead of N+(0, 0.2) (shash_dirichlet_bands_widedelta.stan).
#   epsyear   Canonical model with year-varying skewness
#             epsilon[r,y] = epsilon_r + tau_eps_year * eta[r,y]
#             (shash_dirichlet_bands_epsyear.stan).
#
# Output: ../replicated/sims/sensitivity/<variant>.rds, each a list with $draws (a
# 2,000-row posterior matrix of the kept parameters), $diag, and
# $summary, in the same layout as sims/mi/imp_NN.rds.  The paper reads
# these at render time.
#
# Paths are relative to code/, so run from code/ (bin/generate sims
# does this, after the MI fit):   Rscript run_fit_sensitivity.R
# The three run in parallel; the skew-t is the slow one (15 to 25
# minutes), the other two take a few minutes each.
#
# Not bit-for-bit reproducible: the sampler seed is fixed, but the
# init functions draw from R's RNG inside forked workers that are not
# seeded, so the starting values differ from run to run.  Results
# agree across runs to Monte Carlo error.

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(purrr)
  library(cmdstanr); library(posterior)
  library(parallel)
})

DATA_DIR <- "../replicated/data/race/augmented"
OUT_DIR  <- "../replicated/sims/sensitivity"
MI_DIR   <- "../replicated/sims/mi"
SEED     <- 20260514        # as in run_fit_mi.R
M_REF    <- 1L              # the imputation we reuse
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- data prep: verbatim from run_fit_mi.R --------------------------------
YEARS      <- 2018:2026
MAIN_RACES <- c("White", "Hispanic/Latino", "Black/African American",
                "Asian", "Two or More Races",
                "American Indian/Alaska Native",
                "Native Hawaiian/Other Pacific Islander")
RACES      <- c(MAIN_RACES, "No Response")
NR_IDX     <- which(RACES == "No Response")
BAND_LO    <- c(-Inf, 595, 795, 995, 1195, 1395)
BAND_HI    <- c( 595, 795, 995, 1195, 1395, +Inf)
BAND_ORDER <- c("400-590", "600-790", "800-990",
                "1000-1190", "1200-1390", "1400-1600")

all_data <- map_dfr(YEARS, function(yr)
  read.csv(sprintf("%s/%d.csv", DATA_DIR, yr)) |> mutate(year = yr))

N_arr <- array(0, dim = c(length(RACES), length(YEARS)))
mu_arr <- array(0, dim = c(length(RACES), length(YEARS)))
for (r_i in seq_along(RACES)) for (y_i in seq_along(YEARS)) {
  rec <- subset(all_data,
                year == YEARS[y_i] & race == RACES[r_i] & score_range == "all")
  N_arr[r_i, y_i]  <- rec$test_takers[1]
  mu_arr[r_i, y_i] <- rec$mean_total[1]
}
cohort_counts <- array(0, dim = c(length(YEARS), 6))
for (y_i in seq_along(YEARS)) for (b in seq_len(6)) {
  rec <- subset(all_data,
                year == YEARS[y_i] & race == "Total" & score_range == BAND_ORDER[b])
  cohort_counts[y_i, b] <- rec$test_takers[1]
}
printed_pct <- array(0, dim = c(length(MAIN_RACES), length(YEARS), 6))
for (i in seq_along(MAIN_RACES)) for (y_i in seq_along(YEARS)) for (b in seq_len(6)) {
  rec <- subset(all_data,
                year == YEARS[y_i] & race == MAIN_RACES[i] & score_range == BAND_ORDER[b])
  printed_pct[i, y_i, b] <- rec$percent[1]
}
sample_row <- function(k, max_tries = 100000) {
  los <- pmax(0, (k - 0.5) / 100)
  his <- pmin(1, (k + 0.5) / 100)
  for (try in seq_len(max_tries)) {
    pi <- numeric(6)
    pi[1:5] <- runif(5, los[1:5], his[1:5])
    pi[6]   <- 1 - sum(pi[1:5])
    if (pi[6] >= los[6] && pi[6] < his[6]) return(pi)
  }
  stop("rejection sampling failed for printed_pct = ",
       paste(k, collapse = ","))
}
sample_imputation <- function() {
  p_obs <- array(0, dim = c(length(RACES), length(YEARS), 6))
  for (y_i in seq_along(YEARS)) {
    named_counts <- matrix(0, nrow = length(MAIN_RACES), ncol = 6)
    for (i in seq_along(MAIN_RACES)) {
      r_i <- match(MAIN_RACES[i], RACES)
      pi  <- sample_row(printed_pct[i, y_i, ])
      named_counts[i, ] <- pi * N_arr[r_i, y_i]
      p_obs[r_i, y_i, ] <- pi
    }
    nr_counts <- cohort_counts[y_i, ] - colSums(named_counts)
    nr_counts <- pmax(nr_counts, 1)
    p_obs[NR_IDX, y_i, ] <- nr_counts / sum(nr_counts)
  }
  p_obs
}
gauss_hermite_normal <- function(n) {
  i <- 1:(n - 1)
  J <- matrix(0, n, n)
  J[cbind(i, i + 1)] <- sqrt(i); J[cbind(i + 1, i)] <- sqrt(i)
  eg <- eigen(J, symmetric = TRUE); ord <- order(eg$values)
  list(nodes = eg$values[ord], weights = (eg$vectors[1, ]^2)[ord])
}
G  <- 24L; gq <- gauss_hermite_normal(G)

# ---- the shared imputed dataset (identical to imputation 1) ---------------
set.seed(SEED + M_REF)
p_obs_arr <- sample_imputation()
stan_data <- list(
  R = length(RACES), Y = length(YEARS), B = 6,
  N = N_arr, mu_printed = mu_arr, p_obs = p_obs_arr,
  lo = BAND_LO, hi = BAND_HI,
  year_z = as.numeric(scale(YEARS)),
  prior_mu = 1050, prior_sd = 300,
  G = G, gh_node = gq$nodes, gh_weight = gq$weights,
  nr_idx = NR_IDX
)
saveRDS(list(p_obs = p_obs_arr, RACES = RACES, YEARS = YEARS, N_arr = N_arr,
             m_ref = M_REF, SEED = SEED),
        file.path(OUT_DIR, "meta.rds"), compress = "xz")

# ---- inits ---------------------------------------------------------------
R_n <- length(RACES); Y_n <- length(YEARS)
init_common <- function() list(
  beta_sigma_mean   = log(230) + rnorm(1, 0, 0.05),
  tau_sigma         = abs(rnorm(1, 0.15, 0.03)),
  sigma_log_raw     = rnorm(R_n, 0, 0.3),
  lambda_mean       = rnorm(1, 0.04, 0.01),
  tau_lambda        = abs(rnorm(1, 0.02, 0.005)),
  lambda_raw        = rnorm(R_n, 0, 0.3),
  eta_raw           = matrix(rnorm(R_n * Y_n, 0, 0.3), R_n, Y_n),
  beta_log_phi      = log(280) + rnorm(1, 0, 0.05),
  tau_log_phi       = abs(rnorm(1, 0.1, 0.05)),
  log_phi_raw       = rnorm(R_n, 0, 0.3),
  beta_log_tau_eta  = log(0.02) + rnorm(1, 0, 0.05),
  tau_log_tau_eta   = abs(rnorm(1, 0.1, 0.05)),
  log_tau_eta_raw   = rnorm(R_n, 0, 0.3),
  log_sigma_nr      = log(230) + rnorm(Y_n, 0, 0.05),
  theta_mean        = mu_arr + matrix(rnorm(R_n * Y_n, 0, 5), R_n, Y_n)
)
init_shash <- function() c(init_common(), list(
  beta_eps_mean     = rnorm(1, 0.2, 0.05),
  tau_eps           = abs(rnorm(1, 0.25, 0.05)),
  eps_raw           = rnorm(R_n, 0, 0.3),
  beta_log_delta    = rnorm(1, 0.1, 0.05),
  tau_log_delta     = abs(rnorm(1, 0.05, 0.02)),
  log_delta_raw     = rnorm(R_n, 0, 0.3)
))
init_epsyear <- function() c(init_shash(), list(
  tau_eps_year      = abs(rnorm(1, 0.05, 0.02)),
  eps_year_raw      = matrix(rnorm(R_n * Y_n, 0, 0.3), R_n, Y_n)
))
init_skewt <- function() c(init_common(), list(
  beta_log_gamma       = rnorm(1, 0.2, 0.05),
  tau_log_gamma        = abs(rnorm(1, 0.15, 0.03)),
  log_gamma_raw        = rnorm(R_n, 0, 0.3),
  beta_log_nu_excess   = log(28) + rnorm(1, 0, 0.1),
  tau_log_nu_excess    = abs(rnorm(1, 0.3, 0.05)),
  log_nu_excess_raw    = rnorm(R_n, 0, 0.3)
))

KEEP_COMMON <- c("p_above_1400", "p_above_1450", "p_above_1500",
                 "p_above_1550", "p_above_1600", "theta_mean", "mu",
                 "sigma_ry", "phi", "log_lik")
VARIANTS <- list(
  skewt = list(stan = "skewt_dirichlet_bands.stan", init = init_skewt,
               keep = c(KEEP_COMMON, "gamma_r", "nu_r"),
               # the skew-t posterior is harder to sample (student_t_cdf
               # in the likelihood); settings from the original runner
               adapt_delta = 0.97, max_treedepth = 13),
  widedelta = list(stan = "shash_dirichlet_bands_widedelta.stan", init = init_shash,
                   keep = c(KEEP_COMMON, "epsilon", "delta", "tau_log_delta"),
                   adapt_delta = 0.95, max_treedepth = 10),
  epsyear = list(stan = "shash_dirichlet_bands_epsyear.stan", init = init_epsyear,
                 keep = c(KEEP_COMMON, "epsilon_ry", "delta", "tau_eps_year"),
                 adapt_delta = 0.95, max_treedepth = 10)
)

# compile up front (sequentially) so the parallel fits don't race on it
mods <- lapply(VARIANTS, function(v) cmdstan_model(v$stan))

fit_variant <- function(name) {
  v <- VARIANTS[[name]]
  t0 <- Sys.time()
  fit <- mods[[name]]$sample(
    data = stan_data, chains = 2, parallel_chains = 2,
    iter_warmup = 1000, iter_sampling = 1000,
    adapt_delta = v$adapt_delta, max_treedepth = v$max_treedepth,
    seed = SEED + M_REF, refresh = 0, init = v$init,
    show_messages = FALSE, show_exceptions = FALSE
  )
  draws <- fit$draws(v$keep, format = "matrix")
  out <- list(
    variant = name,
    draws   = matrix(as.numeric(draws), nrow = nrow(draws),
                     dimnames = list(NULL, colnames(draws))),
    diag    = fit$diagnostic_summary(),
    summary = fit$summary(v$keep)[, c("variable", "rhat", "ess_bulk", "ess_tail")],
    secs    = as.numeric(difftime(Sys.time(), t0, units = "secs"))
  )
  saveRDS(out, file.path(OUT_DIR, paste0(name, ".rds")), compress = "xz")
  cat(sprintf("  %-10s done in %.0fs; divergences %d; max rhat %.3f\n",
              name, out$secs, sum(out$diag$num_divergent),
              max(out$summary$rhat, na.rm = TRUE)))
  out
}

cat("Fitting", length(VARIANTS), "sensitivity variants on imputation", M_REF, "\n")
res <- mclapply(names(VARIANTS), fit_variant, mc.cores = length(VARIANTS),
                mc.preschedule = FALSE)
errs <- which(sapply(res, inherits, "try-error"))
if (length(errs)) stop("failed: ", paste(names(VARIANTS)[errs], collapse = ", "))

# headline comparison against the canonical fit on the same imputation
ref <- readRDS(file.path(MI_DIR, "imp_01.rds"))$draws
y25 <- match(max(YEARS), YEARS); ai <- match("Asian", RACES); bi <- match("Black/African American", RACES)
headline <- function(d, thr) {
  counts <- sapply(seq_along(RACES), function(r)
    N_arr[r, y25] * d[, sprintf("p_above_%d[%d,%d]", thr, r, y25)])
  pb <- 100 * counts[, bi] / rowSums(counts)
  sprintf("%.2f%% (%.2f-%.2f)", median(pb), quantile(pb, .025), quantile(pb, .975))
}
cat("\n", max(YEARS), " Black share of the above-threshold pool, imputation 1 only:\n")
cat(sprintf("  %-10s 1500: %s   1550: %s\n", "canonical", headline(ref, 1500), headline(ref, 1550)))
for (o in res)
  cat(sprintf("  %-10s 1500: %s   1550: %s\n", o$variant, headline(o$draws, 1500), headline(o$draws, 1550)))
