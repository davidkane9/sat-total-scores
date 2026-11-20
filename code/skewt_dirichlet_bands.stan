// Two-piece Fernandez-Steel skew-t variant of the SHASH+Dirichlet
// model, kept available as a sensitivity-analysis fit, not the
// canonical model.  Same fitting framework as the canonical SHASH model
// (free per-cell theta_mean with a weakly informative prior, per-race
// scale trend with No-Response free per-year, Dirichlet likelihood on
// observed band proportions, hierarchical priors on per-group shape
// and scale parameters) but the within-group distribution is the
// Fernandez-Steel (1998) two-piece skew-t in place of sinh-arcsinh.
//
// Fernandez-Steel skew-t (gamma > 0, nu > 2):
//   density   f(z) = (2/(gamma + 1/gamma)) * t_nu(z * gamma^{-sign(z)})
//   CDF       F(z) =     (2/(gamma^2+1)) * T_nu(z*gamma)            for z <= 0
//             F(z) = 1/(gamma^2+1) + (2 gamma^2/(gamma^2+1)) *
//                                    [T_nu(z/gamma) - 1/2]            for z > 0
//   mean      E[Z] = (gamma - 1/gamma) * sqrt(nu/pi) *
//                                    Gamma((nu-1)/2)/Gamma(nu/2)
// where t_nu, T_nu are the standard Student-t density and CDF.
//
// Per-group skewness gamma_r and degrees of freedom nu_r are pooled
// hierarchically (on log scales).  All scale and concentration
// parameters share the same hierarchical structure as the canonical
// SHASH model: delta_r is gone (FS has no analogue) and is replaced
// by the per-group nu_r.

functions {
  real fs_skew_t_F(real z, real gamma, real nu) {
    // Numerical guards: Stan's student_t_cdf returns NaN for non-finite
    // arguments, which HMC's leapfrog can occasionally produce during
    // warmup.  Clamp the standardized score to a wide-but-finite range.
    if (is_inf(z) || is_nan(z)) {
      if (z < 0) return 0.0;
      if (z > 0) return 1.0;
      return 0.5;
    }
    real g2 = gamma * gamma;
    if (z <= 0) {
      real zg = z * gamma;
      if (zg < -40) return 0.0;
      return (2.0 / (g2 + 1.0)) * student_t_cdf(zg | nu, 0.0, 1.0);
    } else {
      real zg = z / gamma;
      if (zg > 40) return 1.0;
      return 1.0 / (g2 + 1.0)
             + (2.0 * g2 / (g2 + 1.0)) *
               (student_t_cdf(zg | nu, 0.0, 1.0) - 0.5);
    }
  }
  real fs_skew_t_mean(real gamma, real nu) {
    real M = sqrt(nu / pi()) * exp(lgamma((nu - 1) / 2.0) - lgamma(nu / 2.0));
    return (gamma - 1.0 / gamma) * M;
  }
}

data {
  int<lower=1> R;
  int<lower=1> Y;
  int<lower=1> B;
  array[R, Y]    int<lower=0> N;
  array[R, Y]    real           mu_printed;
  array[R, Y]    vector[B]       p_obs;
  array[B]       real           lo;
  array[B]       real           hi;
  vector[Y]      year_z;
  real           prior_mu;
  real<lower=0>  prior_sd;
  int<lower=1>   G;
  vector[G]      gh_node;       // not used by FS skew-t; kept for driver compatibility
  vector[G]      gh_weight;
  int<lower=1, upper=R> nr_idx;
}

transformed data {
  array[R, Y] vector[B] p_obs_n;
  for (r in 1:R)
    for (y in 1:Y)
      p_obs_n[r, y] = p_obs[r, y] / sum(p_obs[r, y]);
}

parameters {
  real beta_sigma_mean;
  real<lower=0> tau_sigma;
  vector[R] sigma_log_raw;
  real lambda_mean;
  real<lower=0> tau_lambda;
  vector[R] lambda_raw;
  matrix[R, Y] eta_raw;
  // Per-group log-skewness log_gamma_r ~ N(beta_log_gamma, tau_log_gamma)
  real beta_log_gamma;
  real<lower=0> tau_log_gamma;
  vector[R] log_gamma_raw;
  // Per-group log-(nu-2): nu_r = 2 + exp(...) so nu_r > 2 (mean and variance finite)
  real beta_log_nu_excess;
  real<lower=0> tau_log_nu_excess;
  vector[R] log_nu_excess_raw;
  // Per-group Dirichlet concentration phi_r
  real beta_log_phi;
  real<lower=0> tau_log_phi;
  vector[R] log_phi_raw;
  // Per-group wiggle spread tau_eta_r
  real beta_log_tau_eta;
  real<lower=0> tau_log_tau_eta;
  vector[R] log_tau_eta_raw;
  vector[Y] log_sigma_nr;
  array[R, Y] real theta_mean;
}

transformed parameters {
  vector[R] sigma_log0  = beta_sigma_mean + tau_sigma  * sigma_log_raw;
  vector[R] lambda      = lambda_mean      + tau_lambda * lambda_raw;
  vector[R] gamma_r     = exp(beta_log_gamma     + tau_log_gamma     * log_gamma_raw);
  // Floor nu_r at 2.5 to keep both mean and variance comfortably finite
  // and avoid Gamma-ratio overflow in fs_skew_t_mean during warmup.
  vector[R] nu_r;
  for (r in 1:R)
    nu_r[r] = 2.5 + exp(beta_log_nu_excess + tau_log_nu_excess * log_nu_excess_raw[r]);
  vector[R] phi         = exp(beta_log_phi       + tau_log_phi       * log_phi_raw);
  vector[R] tau_eta     = exp(beta_log_tau_eta   + tau_log_tau_eta   * log_tau_eta_raw);

  vector[R] mX;
  for (r in 1:R) mX[r] = fs_skew_t_mean(gamma_r[r], nu_r[r]);

  array[R, Y] real sigma_ry;
  array[R, Y] real mu;
  for (r in 1:R)
    for (y in 1:Y) {
      real raw_log_sigma;
      if (r == nr_idx)
        raw_log_sigma = log_sigma_nr[y];
      else
        raw_log_sigma = sigma_log0[r] + lambda[r] * year_z[y]
                        + tau_eta[r] * eta_raw[r, y];
      // Floor sigma at 1 score point to prevent numerical underflow in
      // student_t_cdf during HMC leapfrog overshoots.
      sigma_ry[r, y] = fmax(exp(raw_log_sigma), 1.0);
      mu[r, y] = theta_mean[r, y] - sigma_ry[r, y] * mX[r];
    }
}

model {
  // Same hyperpriors on scale and trend as SHASH
  beta_sigma_mean ~ normal(log(230), 0.4);
  tau_sigma       ~ normal(0, 0.35);
  lambda_mean     ~ normal(0, 0.15);
  tau_lambda      ~ normal(0, 0.06);

  // Hierarchy on log-skewness gamma_r (centred at gamma = 1, symmetric);
  // bias toward right-skew (mean log_gamma > 0) is mild and weakly informative.
  beta_log_gamma  ~ normal(0.2, 0.4);
  tau_log_gamma   ~ normal(0, 0.3);

  // Hierarchy on log-(nu - 2): centred where nu ~ 30 (near-normal tails),
  // weakly informative so tail can become heavier or lighter if the data demand.
  beta_log_nu_excess ~ normal(log(28), 1.2);
  tau_log_nu_excess  ~ normal(0, 0.6);

  // Hierarchies on phi_r and tau_eta_r match the canonical SHASH model
  beta_log_phi     ~ normal(log(300), 1.5);
  tau_log_phi      ~ normal(0, 0.5);
  beta_log_tau_eta ~ normal(log(0.03), 0.6);
  tau_log_tau_eta  ~ normal(0, 0.3);

  sigma_log_raw      ~ std_normal();
  lambda_raw         ~ std_normal();
  to_vector(eta_raw) ~ std_normal();
  log_gamma_raw      ~ std_normal();
  log_nu_excess_raw  ~ std_normal();
  log_phi_raw        ~ std_normal();
  log_tau_eta_raw    ~ std_normal();

  log_sigma_nr ~ normal(log(230), 0.4);

  for (r in 1:R)
    for (y in 1:Y)
      theta_mean[r, y] ~ normal(prior_mu, prior_sd);

  for (r in 1:R) {
    for (y in 1:Y) {
      vector[B] pi;
      for (b in 1:B) {
        real Fhi = is_inf(hi[b]) ? 1.0
                   : fs_skew_t_F((hi[b] - mu[r, y]) / sigma_ry[r, y], gamma_r[r], nu_r[r]);
        real Flo = is_inf(lo[b]) ? 0.0
                   : fs_skew_t_F((lo[b] - mu[r, y]) / sigma_ry[r, y], gamma_r[r], nu_r[r]);
        pi[b] = fmax(Fhi - Flo, 1e-8);
      }
      pi = pi / sum(pi);
      // Guard against numerical zero in the concentration parameter
      vector[B] alpha_dir = fmax(phi[r] * pi, 1e-8);
      p_obs_n[r, y] ~ dirichlet(alpha_dir);
    }
  }
}

generated quantities {
  array[R, Y] vector[B] p_rep;
  array[R, Y] vector[B] pi_hat;
  array[R, Y] real mean_implied;
  array[R, Y] real p_above_1400;
  array[R, Y] real p_above_1450;
  array[R, Y] real p_above_1500;
  array[R, Y] real p_above_1550;
  array[R, Y] real p_above_1600;
  array[R, Y] real log_lik;

  for (r in 1:R) {
    for (y in 1:Y) {
      real s = sigma_ry[r, y];
      real g = gamma_r[r];
      real v = nu_r[r];
      real m = mu[r, y];
      vector[B] pi;
      for (b in 1:B) {
        real Fhi = is_inf(hi[b]) ? 1.0 : fs_skew_t_F((hi[b] - m) / s, g, v);
        real Flo = is_inf(lo[b]) ? 0.0 : fs_skew_t_F((lo[b] - m) / s, g, v);
        pi[b] = fmax(Fhi - Flo, 1e-12);
      }
      pi = pi / sum(pi);
      pi_hat[r, y]       = pi;
      mean_implied[r, y] = theta_mean[r, y];
      p_rep[r, y]        = dirichlet_rng(phi[r] * pi);
      p_above_1400[r, y] = 1 - fs_skew_t_F((1395 - m) / s, g, v);
      p_above_1450[r, y] = 1 - fs_skew_t_F((1445 - m) / s, g, v);
      p_above_1500[r, y] = 1 - fs_skew_t_F((1495 - m) / s, g, v);
      p_above_1550[r, y] = 1 - fs_skew_t_F((1545 - m) / s, g, v);
      p_above_1600[r, y] = 1 - fs_skew_t_F((1595 - m) / s, g, v);
      log_lik[r, y]      = dirichlet_lpdf(p_obs_n[r, y] | phi[r] * pi);
    }
  }
}
