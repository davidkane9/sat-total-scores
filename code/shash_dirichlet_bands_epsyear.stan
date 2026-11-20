// Variant of shash_dirichlet_bands.stan with YEAR-VARYING skewness
// epsilon[r,y].  Per-group base epsilon[r] is hierarchically pooled
// across groups (as in the canonical fit); per-year deviations
// eps_year[r,y] ~ N(0, tau_eps_year) are pooled across all (r,y) with
// a tight prior on tau_eps_year, so the data can admit year-to-year
// drift in skewness where they support it and the prior shrinks toward
// year-constant where they do not.
//
// Everything else matches shash_dirichlet_bands.stan: per-group
// log-linear scale trend, per-cell scale residual, year-constant
// per-group tail-weight delta_r, per-group Dirichlet phi_r, free
// per-cell location, NR-special scale.

functions {
  real shash_F(real x, real eps, real delta) {
    return Phi(sinh(delta * asinh(x) - eps));
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
  vector[G]      gh_node;
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
  // Per-group base skewness (as before), plus a year-varying residual
  real beta_eps_mean;
  real<lower=0> tau_eps;
  vector[R] eps_raw;
  real<lower=0> tau_eps_year;        // shared year-to-year SD
  matrix[R, Y] eps_year_raw;         // standardised year-level residual
  // Per-group tail-weight delta_r
  real beta_log_delta;
  real<lower=0> tau_log_delta;
  vector[R] log_delta_raw;
  // Per-group Dirichlet concentration phi_r
  real beta_log_phi;
  real<lower=0> tau_log_phi;
  vector[R] log_phi_raw;
  // Per-group per-cell-scale residual SD tau_eta_r
  real beta_log_tau_eta;
  real<lower=0> tau_log_tau_eta;
  vector[R] log_tau_eta_raw;
  vector[Y] log_sigma_nr;
  array[R, Y] real theta_mean;
}

transformed parameters {
  vector[R] sigma_log0 = beta_sigma_mean + tau_sigma  * sigma_log_raw;
  vector[R] lambda     = lambda_mean      + tau_lambda * lambda_raw;
  vector[R] eps_r      = beta_eps_mean    + tau_eps    * eps_raw;
  vector[R] delta      = exp(beta_log_delta    + tau_log_delta    * log_delta_raw);
  vector[R] phi        = exp(beta_log_phi      + tau_log_phi      * log_phi_raw);
  vector[R] tau_eta    = exp(beta_log_tau_eta  + tau_log_tau_eta  * log_tau_eta_raw);

  matrix[R, Y] epsilon_ry;
  for (r in 1:R)
    for (y in 1:Y)
      epsilon_ry[r, y] = eps_r[r] + tau_eps_year * eps_year_raw[r, y];

  // Per-(group, year) mean of standardized SHASH variable X, used to
  // convert theta_mean into the raw location mu.
  array[R, Y] real mX_ry;
  for (r in 1:R)
    for (y in 1:Y) {
      real acc = 0;
      real e = epsilon_ry[r, y];
      real d = delta[r];
      for (g in 1:G)
        acc += gh_weight[g] * sinh((asinh(gh_node[g]) + e) / d);
      mX_ry[r, y] = acc;
    }

  array[R, Y] real sigma_ry;
  array[R, Y] real mu;
  for (r in 1:R)
    for (y in 1:Y) {
      if (r == nr_idx)
        sigma_ry[r, y] = exp(log_sigma_nr[y]);
      else
        sigma_ry[r, y] = exp(sigma_log0[r] + lambda[r] * year_z[y] + tau_eta[r] * eta_raw[r, y]);
      mu[r, y] = theta_mean[r, y] - sigma_ry[r, y] * mX_ry[r, y];
    }
}

model {
  beta_sigma_mean ~ normal(log(230), 0.4);
  tau_sigma       ~ normal(0, 0.35);
  lambda_mean     ~ normal(0, 0.15);
  tau_lambda      ~ normal(0, 0.06);
  beta_eps_mean   ~ normal(0, 0.6);
  tau_eps         ~ normal(0, 0.8);
  tau_eps_year    ~ normal(0, 0.15);        // weak: typical year-to-year drift well below the cross-group spread

  beta_log_delta   ~ normal(0, 0.3);
  tau_log_delta    ~ normal(0, 0.2);

  beta_log_phi     ~ normal(log(300), 1.5);
  tau_log_phi      ~ normal(0, 0.5);

  beta_log_tau_eta ~ normal(log(0.03), 0.6);
  tau_log_tau_eta  ~ normal(0, 0.3);

  sigma_log_raw         ~ std_normal();
  lambda_raw            ~ std_normal();
  to_vector(eta_raw)    ~ std_normal();
  eps_raw               ~ std_normal();
  to_vector(eps_year_raw) ~ std_normal();
  log_delta_raw         ~ std_normal();
  log_phi_raw           ~ std_normal();
  log_tau_eta_raw       ~ std_normal();

  log_sigma_nr ~ normal(log(230), 0.4);

  for (r in 1:R)
    for (y in 1:Y)
      theta_mean[r, y] ~ normal(prior_mu, prior_sd);

  for (r in 1:R) {
    for (y in 1:Y) {
      real e = epsilon_ry[r, y];
      real d = delta[r];
      real s = sigma_ry[r, y];
      real m = mu[r, y];
      vector[B] pi;
      for (b in 1:B) {
        real Fhi = is_inf(hi[b]) ? 1.0
                   : shash_F((hi[b] - m) / s, e, d);
        real Flo = is_inf(lo[b]) ? 0.0
                   : shash_F((lo[b] - m) / s, e, d);
        pi[b] = fmax(Fhi - Flo, 1e-12);
      }
      pi = pi / sum(pi);
      p_obs_n[r, y] ~ dirichlet(phi[r] * pi);
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
      real e = epsilon_ry[r, y];
      real d = delta[r];
      real m = mu[r, y];
      vector[B] pi;
      for (b in 1:B) {
        real Fhi = is_inf(hi[b]) ? 1.0 : shash_F((hi[b] - m) / s, e, d);
        real Flo = is_inf(lo[b]) ? 0.0 : shash_F((lo[b] - m) / s, e, d);
        pi[b] = fmax(Fhi - Flo, 1e-12);
      }
      pi = pi / sum(pi);
      pi_hat[r, y]       = pi;
      mean_implied[r, y] = theta_mean[r, y];
      p_rep[r, y]        = dirichlet_rng(phi[r] * pi);
      p_above_1400[r, y] = 1 - shash_F((1395 - m) / s, e, d);
      p_above_1450[r, y] = 1 - shash_F((1445 - m) / s, e, d);
      p_above_1500[r, y] = 1 - shash_F((1495 - m) / s, e, d);
      p_above_1550[r, y] = 1 - shash_F((1545 - m) / s, e, d);
      p_above_1600[r, y] = 1 - shash_F((1595 - m) / s, e, d);
      log_lik[r, y]      = dirichlet_lpdf(p_obs_n[r, y] | phi[r] * pi);
    }
  }
}
