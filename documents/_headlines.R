# Single source of truth for the headline numbers quoted in
# sat_model.qmd, which sources this script in its setup chunk.
#
# Exposes these variables in the caller's environment.  Percentages
# are pre-formatted strings with one decimal (so "1.0" does not render
# as "1"); counts are formatted with thousands separators; ratios are
# integers.
#
#   cohort_n_h, cohort_mean_h, cohort_sd_h   -- headline-year cohort size, published
#                                               mean and SD (College Board)
#   mean_h_asian, mean_h_white, mean_h_black -- headline-year published group means
#   wb_gap_h, wb_d_h, wb_sd15_h             -- headline-year White minus Black mean
#                                               gap in points; as Cohen's d
#                                               against the published cohort
#                                               SD; and on a 15-point-SD metric
#   blk_1400_pct, asn_blk_1400               -- headline-year Black share (%) of the
#                                               >=1400 pool and Asian/Black
#                                               count ratio, data-pinned
#   blk_<t>_med/lo/hi, ab_<t>_med/lo/hi      -- for t in 1500, 1550, 1580:
#                                               headline-year Black share (%) of the
#                                               >=t pool and the Asian/Black
#                                               count ratio, posterior median
#                                               and 95% credible interval
#   nr_1500_pct                              -- headline-year No Response share (%) of
#                                               the >=1500 pool (median)
#   tail_tab                                 -- data frame: for t in 1500,
#                                               1550, 1580 and group in
#                                               Cohort, Asian, White, Black:
#                                               share of that group scoring
#                                               >= t (%) and the implied
#                                               count, medians and 95% CIs
#   tq(t, group, what)                       -- accessor returning the
#                                               formatted string from tail_tab
#   p_above_mat(t, year)                     -- draws x 8 matrix of
#                                               P(score >= t) per group
#   shash_F                                  -- the SHASH CDF

# Load the MI posterior + CSV-derived data arrays
# (.draws_dm, .RACES, .YEARS, .N_arr, .P_OBS, .mu_arr, .p_obs).
source("_inputs.R")

shash_F <- function(x, eps, delta) pnorm(sinh(delta * asinh(x) - eps))

HEADLINE_YEAR <- max(.YEARS)
.yH <- match(HEADLINE_YEAR, .YEARS)
.ai  <- match("Asian", .RACES)
.wi  <- match("White", .RACES)
.bi  <- match("Black/African American", .RACES)
.ni  <- match("No Response", .RACES)

.fmt_pct <- function(x) sprintf("%.1f", x)
.fmt_n   <- function(x) format(round(x), big.mark = ",", trim = TRUE)

# ----- published cohort and group numbers (no model) ------------------------
.coh <- read.csv("../replicated/data/moments/cohort.csv")
.coh_h <- .coh[.coh$year == HEADLINE_YEAR, ]
cohort_n_h    <- .fmt_n(sum(.N_arr[, .yH]))
cohort_mean_h <- .coh_h$mean
cohort_sd_h   <- if (is.na(.coh_h$sd)) NA_real_ else .coh_h$sd
mean_h_asian  <- .mu_arr[.ai, .yH]
mean_h_white  <- .mu_arr[.wi, .yH]
mean_h_black  <- .mu_arr[.bi, .yH]
wb_gap_h      <- mean_h_white - mean_h_black
wb_d_h        <- sprintf("%.2f", wb_gap_h / cohort_sd_h)
wb_sd15_h    <- round(15 * wb_gap_h / cohort_sd_h)

# ----- P(score >= t) per group, from the posterior -------------------------
# For the thresholds saved by the Stan program (1400, 1450, 1500, 1550,
# 1600) use the generated quantities directly; for any other threshold
# evaluate the SHASH survival from the per-cell draws.  A reported score
# >= t corresponds to a latent score >= t - 5 (10-point grid).
p_above_mat <- function(t, year = HEADLINE_YEAR) {
  yi <- match(year, .YEARS)
  saved <- sprintf("p_above_%d[%d,%d]", t, seq_along(.RACES), yi)
  if (all(saved %in% colnames(.draws_dm)))
    return(.draws_dm[, saved, drop = FALSE])
  sapply(seq_along(.RACES), function(r) {
    m <- .draws_dm[, sprintf("mu[%d,%d]", r, yi)]
    s <- .draws_dm[, sprintf("sigma_ry[%d,%d]", r, yi)]
    e <- .draws_dm[, sprintf("epsilon[%d]", r)]
    d <- .draws_dm[, sprintf("delta[%d]", r)]
    1 - shash_F((t - 5 - m) / s, e, d)
  })
}

.counts_above <- function(t) {
  p <- p_above_mat(t)
  sweep(p, 2, .N_arr[, .yH], `*`)
}

.q  <- function(x, p) unname(quantile(x, p))

for (.t in c(1500, 1550, 1580)) {
  .cts <- .counts_above(.t)
  .blk <- 100 * .cts[, .bi] / rowSums(.cts)
  .ab  <- .cts[, .ai] / .cts[, .bi]
  assign(sprintf("blk_%d_med", .t), .fmt_pct(median(.blk)))
  assign(sprintf("blk_%d_lo",  .t), .fmt_pct(.q(.blk, 0.025)))
  assign(sprintf("blk_%d_hi",  .t), .fmt_pct(.q(.blk, 0.975)))
  assign(sprintf("ab_%d_med",  .t), round(median(.ab)))
  assign(sprintf("ab_%d_lo",   .t), round(.q(.ab, 0.025)))
  assign(sprintf("ab_%d_hi",   .t), round(.q(.ab, 0.975)))
}
.cts <- .counts_above(1500)
nr_1500_pct <- sprintf("%.0f", median(100 * .cts[, .ni] / rowSums(.cts)))
# Black share of the >=1500 pool with No Response removed from the
# denominator, i.e. among test-takers who reported a race/ethnicity.
.blk_named <- 100 * .cts[, .bi] / rowSums(.cts[, -.ni])
blk_1500_named_med <- .fmt_pct(median(.blk_named))
blk_1500_named_lo  <- .fmt_pct(.q(.blk_named, 0.025))
blk_1500_named_hi  <- .fmt_pct(.q(.blk_named, 0.975))

# ----- table of shares and counts above each threshold ----------------------
.GROUPS <- c(Cohort = NA, Asian = .ai, White = .wi, Black = .bi)
tail_tab <- do.call(rbind, lapply(c(1500, 1550, 1580), function(t) {
  cts <- .counts_above(t)
  do.call(rbind, lapply(names(.GROUPS), function(g) {
    r <- .GROUPS[[g]]
    n   <- if (is.na(r)) rowSums(cts) else cts[, r]
    den <- if (is.na(r)) sum(.N_arr[, .yH]) else .N_arr[r, .yH]
    pct <- 100 * n / den
    data.frame(threshold = t, group = g,
               pct_med = .fmt_pct(median(pct)),
               pct_lo  = .fmt_pct(.q(pct, 0.025)),
               pct_hi  = .fmt_pct(.q(pct, 0.975)),
               n_med   = .fmt_n(median(n)),
               n_lo    = .fmt_n(.q(n, 0.025)),
               n_hi    = .fmt_n(.q(n, 0.975)),
               stringsAsFactors = FALSE)
  }))
}))
tq <- function(t, group, what)
  tail_tab[tail_tab$threshold == t & tail_tab$group == group, what, drop = TRUE]

# ----- headline-year 1400+ data-pinned quantities (no model) -------------------------
.n_1400      <- .N_arr[, .yH] * .P_OBS[, .yH, 6]
blk_1400_pct <- .fmt_pct(100 * .n_1400[.bi] / sum(.n_1400))
asn_blk_1400 <- round(.n_1400[.ai] / .n_1400[.bi])

rm(.t, .cts, .blk, .ab, .coh, .coh_h)
