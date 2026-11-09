"""Lorenz-96 ensemble assimilation with LESRF, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (LESRF) analysis update for Lorenz-96.
    Uses adaptive multiplicative inflation and Gaussian localization.
    
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
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target spread: balance between observation noise and model uncertainty
    target_spread = np.sqrt(R_val) * 1.4
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    ratio = target_spread / np.maximum(current_spread, 1e-10)
    inflation = np.clip(np.mean(ratio), 0.95, 1.2)
    
    # Blend with a base inflation for stability
    base_inflation = 1.04
    inflation = 0.5 * base_inflation + 0.5 * inflation
    inflation = np.clip(inflation, 1.0, 1.2)
    
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # --- Localization ---
    # Gaussian localization with sigma tuned for Lorenz-96 correlation length
    sigma = 3.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (n_vars, n_obs)
    
    # --- LESRF Update ---
    # Step 1: Compute observation-space covariance
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (n_obs, n_obs)
    S = C_oo + R_val * np.eye(n_obs)
    
    # Step 2: Compute cross-covariance with localization
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (n_vars, n_obs)
    C_xo_loc = C_xo * loc_weights
    
    # Step 3: Compute gain using Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L.T, C_xo_loc.T)
        Y = np.linalg.solve(L, X)
        K = Y.T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # Step 4: Apply update to each member
    # x_i^a = x_i^f + K @ (obs - x_i^f[obs_idx])
    innovations = obs[None, :] - ens[:, obs_idx]  # Shape (n_members, n_obs)
    K_innov = innovations @ K.T  # Shape (n_members, n_vars)
    ens = ens + K_innov
    
    # Clip and check for finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
