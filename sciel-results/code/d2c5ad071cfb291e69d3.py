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
    
    # Target spread: slightly larger than observation std to maintain spread
    target_spread = np.sqrt(R_val) * 1.4
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    
    # Inflation factor based on spread ratio
    ratio = target_spread / np.maximum(current_spread, 1e-12)
    inflation = np.clip(np.mean(ratio), 0.95, 1.18)
    # Blend with a small base inflation for stability
    inflation = 0.6 * 1.02 + 0.4 * inflation
    inflation = np.clip(inflation, 1.0, 1.15)
    
    # Apply inflation
    ens = ens_mean + inflation * ens_centered
    
    # Recompute mean after inflation (should be same but be safe)
    ens_mean = np.mean(ens, axis=0)
    
    # --- Localization weights ---
    # Gaussian localization with sigma tuned for Lorenz-96
    sigma = 3.0
    indices = np.arange(n_vars)
    # Circular distance
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Localization weights: (n_vars, n_obs)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- LESRF Analysis ---
    # For each member, update the observed components using a local Kalman filter
    # This is the square-root form that avoids computing the full covariance matrix
    
    # Precompute the observation-space quantities
    # A_obs: (n_members, n_obs) - anomalies at observed locations
    A_obs = ens[:, obs_idx] - ens_mean[obs_idx]
    
    # Observation covariance (10x10) - but we work in square-root form
    # The innovation covariance: S = P_oo + R
    # P_oo = (1/(N-1)) * A_obs^T @ A_obs
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky decomposition of S
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Add jitter if not positive definite
        jitter = 1e-8
        S_reg = S + jitter * np.eye(n_obs)
        L = np.linalg.cholesky(S_reg)
    
    # Innovation: (n_obs,)
    innovation = obs - ens_mean[obs_idx]
    
    # Solve L @ y = innovation for y
    y = np.linalg.solve(L, innovation)
    
    # For each member, compute the local update
    # The gain in square-root form: K = P_xo @ S^{-1}
    # But in LESRF, we update each member's observed components directly
    
    # Cross-covariance P_xo: (n_vars, n_obs)
    A_full = ens - ens_mean  # (n_members, n_vars)
    P_xo = (A_full.T @ A_obs) / (n_members - 1.0)
    
    # Apply localization to cross-covariance
    P_xo_loc = P_xo * loc_weights
    
    # Gain: K = P_xo_loc @ S^{-1}, shape (n_vars, n_obs)
    # S^{-1} can be computed via L: S^{-1} = L^{-T} @ L^{-1}
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Update mean
    delta_mean = K @ innovation
    ens_mean_new = ens_mean + delta_mean
    
    # Update anomalies
    # A_new = A - A_obs @ K.T
    A_obs_Kt = A_obs @ K.T  # (n_members, n_vars)
    A_new = A_full - A_obs_Kt
    
    # Reconstruct ensemble
    ens = ens_mean_new[None, :] + A_new
    
    # --- Post-processing ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
