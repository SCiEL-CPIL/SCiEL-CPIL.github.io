"""Lorenz-96 ensemble assimilation with Stochastic EnKF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation and Gaussian localization with
    observation perturbation to preserve ensemble spread.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)
    
    # --- Adaptive Inflation ---
    # Use innovation-based adaptive inflation for better calibration
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    # Innovation-based adaptive inflation
    innovation = obs - ens_mean[obs_idx]
    P_yy = np.mean(ens_anom[:, obs_idx] ** 2, axis=0)
    expected_innov_var = np.mean(P_yy + R_val)
    innovation_var = np.mean(innovation ** 2)
    
    if expected_innov_var > 1e-10:
        ratio = innovation_var / expected_innov_var
        # Smooth adaptive inflation
        inflation_adaptive = np.clip(ratio, 0.9, 1.3)
        # Blend with spread-based inflation
        target_spread = np.sqrt(R_val) * 1.2
        spread_ratio = target_spread / np.maximum(current_spread, 1e-12)
        inflation_spread = np.clip(np.mean(spread_ratio), 0.95, 1.10)
        inflation_factor = 0.6 * inflation_adaptive + 0.4 * inflation_spread
    else:
        inflation_factor = 1.02
    
    # Blend with small base inflation for stability
    inflation_factor = 0.3 * 1.01 + 0.7 * inflation_factor
    inflation_factor = np.clip(inflation_factor, 1.0, 1.08)
    
    # Apply inflation
    ens = ens_mean + inflation_factor * ens_anom
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean
    
    # --- Localization weights ---
    sigma = 2.5  # Localization length scale
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # --- Observation Perturbation EnKF (Stochastic Observation Noise) ---
    # Anomalies at observed locations
    A_obs = ens_anom[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance between all variables and observations
    P_xo = (ens_anom.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization
    P_xo_loc = P_xo * loc_weights
    
    # Observation error variance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky of S with jitter for stability
    try:
        L = np.linalg.cholesky(S + 1e-8 * np.eye(n_obs))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(S + 1e-6 * np.eye(n_obs))
    
    # Compute gain matrix K = P_xo_loc @ S^{-1}
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv  # (n_obs, n_obs)
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Generate perturbed observations for each ensemble member
    rng = np.random.default_rng(42)
    eps = rng.standard_normal((n_members, n_obs))
    perturbed_obs = obs[None, :] + np.sqrt(R_val) * eps
    
    # For each member, compute innovation: y_i - H*x_f_i
    hx_f = ens_mean[obs_idx][None, :] + A_obs
    member_innovations = perturbed_obs - hx_f
    
    # Update: x_a_i = x_f_i + K @ (y_i - H*x_f_i)
    ens = ens + (K @ member_innovations.T).T
    
    # --- Post-processing: ensure finite and bounded ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
