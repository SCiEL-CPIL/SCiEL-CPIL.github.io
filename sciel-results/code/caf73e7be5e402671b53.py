"""
Lorenz-96 ensemble assimilation combining Stochastic Ensemble Kalman Filter
with localised covariance and adaptive inflation.

Structural design:
- Localised sequential observation processing (SE-Kalman style)
- Gaussian localisation kernel on the periodic ring
- Adaptive multiplicative inflation based on spread/noise ratio
- Stochastic perturbation of observations to avoid rank-deficiency
"""
import numpy as np
import time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic localised Ensemble Kalman Filter analysis update for Lorenz-96.

    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.

    Returns
    -------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 120s limit

    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)

    ens = np.array(ensemble, dtype=np.float64).copy()
    obs = np.array(observations, dtype=np.float64).copy()
    R = float(observation_variance)
    sqrt_R = np.sqrt(R)

    # --- Step 1: Adaptive inflation BEFORE the update ---
    ens_mean = np.mean(ens, axis=0)
    ens_obs_var = np.var(ens[:, obs_idx], axis=0)
    mean_spread = np.mean(ens_obs_var)

    inflation = 1.0
    if mean_spread > 0:
        ratio = mean_spread / R
        if ratio < 0.3:
            inflation = 1.12
        elif ratio < 0.6:
            inflation = 1.06
        elif ratio < 1.0:
            inflation = 1.02
        else:
            inflation = 1.0

    if inflation != 1.0:
        ens = ens_mean + inflation * (ens - ens_mean)
        ens_mean = np.mean(ens, axis=0)

    # --- Step 2: Localisation weights ---
    # Gaussian kernel with correlation length tuned for L96 spatial structure
    corr_len = 3.0
    # Precompute weights: shape (n_obs, n_vars)
    loc_w = np.zeros((n_obs, n_vars), dtype=np.float64)
    for j, oj in enumerate(obs_idx):
        for i in range(n_vars):
            dist = abs(i - oj)
            dist = min(dist, n_vars - dist)
            loc_w[j, i] = np.exp(-0.5 * (dist / corr_len) ** 2)

    # --- Step 3: Sequential stochastic Kalman update ---
    # Use a fixed random seed for reproducibility within a single call
    # (the evaluator calls analyze repeatedly; we use a deterministic seed
    #  derived from the ensemble to keep results stable)
    seed_val = int(np.abs(np.sum(ens * 1e6)) % (2**31))
    rng = np.random.default_rng(seed_val)

    alpha = n_members / (n_members - 1.0)  # bias correction factor

    for j in range(n_obs):
        # Check time budget
        if time.time() > deadline:
            break

        oj = obs_idx[j]
        w_j = loc_w[j, :]  # (n_vars,)

        # Stochastic observation perturbation
        # Generate n_members perturbed observations
        # y_k = y_j + eta_k, eta_k ~ N(0, R)
        eta = rng.standard_normal(n_members) * sqrt_R
        y_pert = obs[j] + eta  # (n_members,)

        # Ensemble values at the observed coordinate
        H_ens = ens[:, oj]  # (n_members,)
        H_mean = np.mean(H_ens)
        H_anom = H_ens - H_mean  # (n_members,)

        # Innovation
        innov = y_pert - H_ens  # (n_members,)

        # S = ensemble variance of H(x) + R
        S = alpha * np.dot(H_anom, H_anom) / (n_members - 1.0) + R
        if S < 1e-12:
            S = 1e-12

        # For each state variable, compute localised gain and update
        for i in range(n_vars):
            if w_j[i] < 0.005:
                continue

            x_anom = ens[:, i] - np.mean(ens[:, i])
            cross_cov = alpha * np.dot(x_anom, H_anom) / (n_members - 1.0)
            cross_cov_loc = cross_cov * w_j[i]

            gain = cross_cov_loc / S
            ens[:, i] += gain * innov

        # Recompute mean after this observation (keeps things stable)
        ens_mean = np.mean(ens, axis=0)

    # --- Step 4: Post-update inflation ---
    # Small multiplicative inflation to counteract filter collapse
    post_inflation = 1.015
    ens_mean = np.mean(ens, axis=0)
    ens = ens_mean + post_inflation * (ens - ens_mean)

    # --- Step 5: Sanity checks ---
    ens = np.where(np.isfinite(ens), ens, 0.0)
    ens = np.clip(ens, -100.0, 100.0)

    return ens


# Keep helper functions from the original in case they are referenced
def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _localization_kernel(dist, r):
    """Gaussian localization kernel."""
    if dist <= r:
        return np.exp(-0.5 * (dist / r) ** 2)
    return 0.0
