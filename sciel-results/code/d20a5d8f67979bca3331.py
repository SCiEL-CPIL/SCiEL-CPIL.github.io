"""Lorenz-96 ensemble assimilation with stochastic EnKF (observation perturbation), adaptive spread correction, and tight localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (observation-perturbation variant) for Lorenz-96.
    
    Key design choices:
      * Observation-perturbation EnKF: adds independent Gaussian noise to each
        member's observation vector, preserving the ensemble covariance structure
        without requiring explicit multiplicative inflation (which can introduce
        bias).
      * Tight Gaussian localization on circular distance for Lorenz-96's
        short-range correlation structure.
      * Mild adaptive spread correction only when ensemble variance at observed
        coordinates is significantly below the observation noise standard deviation,
        to prevent filter collapse without over-inflating.
      * Deterministic gain via SVD for numerical stability with small ensembles.
    
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
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # --- Mild adaptive spread correction (only if under-dispersed) ---
    ens_mean = np.mean(ens, axis=0)
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    ens_std_obs = np.sqrt(np.maximum(ens_var_obs, 1e-12))
    obs_std = np.sqrt(R_val)
    
    # Only inflate if ensemble spread is well below observation noise std
    spread_deficit = np.mean(np.maximum(obs_std - ens_std_obs, 0.0) / np.maximum(obs_std, 1e-12))
    if spread_deficit > 0.1:
        inflation = 1.0 + 0.3 * spread_deficit
        inflation = min(inflation, 1.08)
        ens_centered = ens - ens_mean
        ens = ens_mean + inflation * ens_centered
        ens_mean = np.mean(ens, axis=0)
    
    # --- Localization (tight Gaussian, circular distance) ---
    sigma = 2.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)
    
    # --- Stochastic EnKF: perturb observations per member ---
    # Each member gets its own perturbed observation: y_m = y + eps_m, eps_m ~ N(0, R)
    rng = np.random.default_rng()
    obs_noise = rng.standard_normal((n_members, n_obs)) * np.sqrt(R_val)
    y_perturbed = obs[None, :] + obs_noise  # (n_members, n_obs)
    
    # Compute anomalies
    X_anom = ens - ens_mean  # (n_members, n_vars)
    Y_anom = ens[:, obs_idx] - ens_mean[obs_idx]  # (n_members, n_obs)
    
    # Perturbed observation anomalies (relative to mean observation)
    Y_pert_anom = y_perturbed - obs[None, :]  # (n_members, n_obs), this is just obs_noise
    # Actually for observation perturbation EnKF, we use:
    #   y_m = y + epsilon_m
    # and the innovation for member m is y_m - y_m^ens where y_m^ens = y_ens + epsilon_m
    # So innovation_m = (y + eps_m) - (y_ens + eps_m) = y - y_ens (deterministic!)
    # Wait, that's not right. Let me reconsider.
    #
    # Standard observation-perturbation EnKF:
    #   y_m = y + eps_m,  eps_m ~ N(0, R)
    #   y_m^ens = x_m^obs (the observed component of member m)
    #   innovation_m = y_m - y_m^ens
    #   x_m = x_m + K * innovation_m
    #
    # Where K is computed from ensemble covariances (same for all members).
    # The key property: E[x_m] = x_mean + K*(y - y_ens_mean)  (mean is updated correctly)
    # Var(x_m) = (I-K)*P_ee*(I-K)^T + K*P_yy*K^T  where P_yy includes R
    # This naturally preserves the correct posterior covariance.
    
    # Innovation per member: y_m - ens[:, obs_idx]
    innovation_per_member = y_perturbed - ens[:, obs_idx]  # (n_members, n_obs)
    
    # Compute gain matrix using ensemble covariances
    # P_ee = (X_anom^T X_anom) / (n_members - 1)  -- not needed directly
    # P_yy = (Y_anom^T Y_anom) / (n_members - 1) + R*I
    # P_ey = (X_anom^T Y_anom) / (n_members - 1)
    # K = P_ey * inv(P_yy)
    
    Y_anom = ens[:, obs_idx] - ens_mean[obs_idx]  # (n_members, n_obs)
    
    P_ee_obs = (Y_anom.T @ Y_anom) / (n_members - 1.0)  # (n_obs, n_obs)
    P_yy = P_ee_obs + R_val * np.eye(n_obs)
    
    P_ey = (X_anom.T @ Y_anom) / (n_members - 1.0)  # (n_vars, n_obs)
    
    # Apply localization to cross-covariance
    P_ey_loc = P_ey * loc_weights  # (n_vars, n_obs)
    
    # Compute gain via SVD for stability
    try:
        U, s, Vt = np.linalg.svd(P_yy)
        s_inv = 1.0 / np.maximum(s, 1e-10)
        P_yy_inv = (Vt.T * s_inv) @ U.T
        K = P_ey_loc @ P_yy_inv  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use Cholesky
        try:
            L = np.linalg.cholesky(P_yy + 1e-8 * np.eye(n_obs))
            L_inv = np.linalg.inv(L)
            P_yy_inv = L_inv.T @ L_inv
            K = P_ey_loc @ P_yy_inv
        except np.linalg.LinAlgError:
            # Last resort: ridge regression
            reg = 1e-6 * np.eye(n_obs)
            P_yy_reg = P_yy + reg
            P_yy_inv = np.linalg.inv(P_yy_reg)
            K = P_ey_loc @ P_yy_inv
    
    # Update each member
    # x_m_new = x_m + K * (y_m - x_m^obs)
    ens_new = ens + innovation_per_member @ K.T  # (n_members, n_vars)
    
    # --- Post-processing ---
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)
    
    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)
    
    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
