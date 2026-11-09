"""Lorenz-96 ensemble assimilation: Stochastic EnKF with adaptive inflation and refined localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update for Lorenz-96.
    Uses observation perturbation with adaptive spread correction and
    Gaussian localization for periodic boundary conditions.
    
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
    deadline = start_time + 115.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Step 1: Adaptive spread correction (mild multiplicative inflation) ---
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # Compute current spread at observed locations
    ens_var_obs = np.var(ens[:, obs_idx], axis=0, ddof=1)
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-15))
    
    # Target: spread should be at least comparable to observation noise
    # Use a gentle target to avoid over-inflation
    target_spread = np.sqrt(R_val) * 1.1
    ratio = target_spread / np.maximum(current_spread, 1e-15)
    raw_inflation = np.clip(np.mean(ratio), 1.0, 1.15)
    
    # Only inflate if spread is too small; never deflate
    inflation = max(1.0, raw_inflation)
    
    # Apply multiplicative inflation to anomalies
    ens = ens_mean + inflation * A
    
    # Recompute after inflation
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # --- Step 2: Localization weights (Gaussian, periodic) ---
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    # Narrower sigma for better correlation structure matching
    sigma = 2.0
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Step 3: Observation perturbation (stochastic EnKF) ---
    # Deterministic seed from ensemble + observations for reproducibility
    seed = hash((tuple(ens.ravel()), tuple(obs.ravel()))) % (2**32)
    rng = np.random.default_rng(seed)
    
    R_std = np.sqrt(R_val)
    perturbed_obs = obs[None, :] + R_std * rng.standard_normal((n_members, n_obs))
    
    # --- Step 4: Compute cross-covariance and observation covariance ---
    A_obs = A[:, obs_idx]  # (n_members, n_obs)
    
    # Cross-covariance: P_xo (state to observation)
    P_xo = (A.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization
    P_xo_loc = P_xo * loc_weights
    
    # Observation-space covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Ensure positive definiteness via eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-8:
        S = S + (1e-8 - min_eig) * np.eye(n_obs)
    
    # Solve S^{-1} @ P_xo_loc via Cholesky
    try:
        L = np.linalg.cholesky(S)
        # Solve L @ L^T @ X = P_xo_loc
        L_inv = np.linalg.inv(L)
        S_inv = L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        S_inv = np.linalg.inv(S)
    
    # Kalman gain
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # --- Step 5: Stochastic update per member ---
    ens_obs = ens[:, obs_idx]  # (n_members, n_obs)
    innovation_per_member = perturbed_obs - ens_obs  # (n_members, n_obs)
    
    updates = innovation_per_member @ K.T  # (n_members, n_vars)
    ens_updated = ens + updates
    
    # --- Step 6: Post-processing ---
    ens_updated = np.clip(ens_updated, -100.0, 100.0)
    ens_updated = np.where(np.isfinite(ens_updated), ens_updated, 0.0)
    
    if time.time() > deadline:
        return ens_updated.astype(ensemble.dtype)
    
    return ens_updated.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
