"""Ensemble Square Root Filter with bulk regression for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Square Root Filter (EnSRF) analysis update using
    bulk regression of unobserved coordinates onto observed coordinates.
    
    The update:
    1. Applies a Kalman-type gain to each observed coordinate.
    2. Regresses each unobserved coordinate onto the observed coordinates
       using ensemble covariance, and applies the regression-based correction.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed coordinates (even-indexed).
    observed_indices : array-like
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ens = ensemble.copy().astype(np.float64)
    n_ens, n_state = ens.shape
    
    obs_idx_arr = np.asarray(observed_indices, dtype=int)
    obs = np.asarray(observations, dtype=np.float64)
    R = float(observation_variance)
    n_obs = len(obs_idx_arr)
    
    # Build set for quick lookup
    obs_idx_set = set(obs_idx_arr.tolist())
    
    # Keep forecast ensemble for reference
    X_f = ens.copy()
    
    # Compute anomalies for observed coordinates in forecast
    obs_f = X_f[:, obs_idx_arr]  # (n_ens, n_obs)
    mu_obs_f = np.mean(obs_f, axis=0)  # (n_obs,)
    A_obs_f = obs_f - mu_obs_f[np.newaxis, :]  # (n_ens, n_obs)
    
    # Compute ensemble covariance of observed coordinates
    # C_oo = A_obs_f.T @ A_obs_f / (n_ens - 1), shape (n_obs, n_obs)
    C_oo = (A_obs_f.T @ A_obs_f) / (n_ens - 1)
    
    # For numerical stability, add small regularization
    C_oo_reg = C_oo + 1e-6 * np.eye(n_obs)
    
    # Compute the Kalman gain for the full system
    # S = C_oo + R * I, shape (n_obs, n_obs)
    S = C_oo + R * np.eye(n_obs)
    
    # K = C_oo @ S^{-1}, shape (n_obs, n_obs)
    # We need K for observed coordinates and cross-covariance for unobserved
    
    # Compute S inverse once
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain for observed coordinates: K_obs = C_oo @ S_inv
    K_obs = C_oo @ S_inv  # (n_obs, n_obs)
    
    # Innovation: y - mu_obs_f, shape (n_obs,)
    innovation = obs - mu_obs_f
    
    # Update observed coordinates
    # New mean for observed: mu_obs_a = mu_obs_f + K_obs @ innovation
    mu_obs_a = mu_obs_f + K_obs @ innovation  # (n_obs,)
    
    # Scale factor for observed anomalies
    # The analysis anomaly for observed coords: A_obs_a = A_obs_f @ (I - K_obs)
    # Because: x_obs_a = x_obs_f + K_obs @ (y - mu_obs_f)
    #          = mu_obs_f + A_obs_f + K_obs @ (y - mu_obs_f)
    #          = mu_obs_a + A_obs_f + K_obs @ (A_obs_f - A_obs_f)  ... 
    # Actually: x_obs_a[i] = x_obs_f[i] + K_obs @ (y - H @ x_f)
    # In anomaly form: A_obs_a = A_obs_f + K_obs @ (y - H @ x_f - (y - H @ mu_f))
    #                  A_obs_a = A_obs_f + K_obs @ (H @ A_obs_f - H @ A_obs_f)  ... no
    # Let me be more careful.
    # x_f = mu_f + A_f
    # y = H @ x_f + v = H @ mu_f + H @ A_f + v
    # Innovation: y - H @ mu_f = H @ A_f + v
    # But we use the ensemble mean, so: innovation = y - H @ mu_f
    # x_a = x_f + K @ (y - H @ x_f)
    #     = x_f + K @ (y - H @ mu_f - H @ A_f)
    #     = x_f + K @ (innovation - H @ A_f)
    # For observed coords, H is just a selection matrix.
    # A_a = A_f + K @ (innovation - H @ A_f)
    #     = A_f + K @ innovation - K @ H @ A_f
    # Since H @ A_f = A_obs_f for observed coords:
    # A_obs_a = A_obs_f + K_obs @ innovation - K_obs @ A_obs_f
    #         = A_obs_f @ (I - K_obs) + K_obs @ innovation
    # Wait, that's not right dimensionally. Let me redo.
    # x_a = x_f + K @ (y - H x_f)
    # For a single observed coordinate j with H being a row vector e_j^T:
    # x_j_a = x_j_f + K_j * (y_j - x_j_f) where K_j = var_j/(var_j+R)
    # In vector form for all observed:
    # x_obs_a = x_obs_f + K_obs @ (y - x_obs_f)  ... but K_obs is (n_obs, n_obs)
    # Actually for scalar obs: K_j is scalar. For multiple obs:
    # x_obs_a = x_obs_f + K @ (y - x_obs_f) where K is (n_obs, n_obs)
    # A_obs_a = A_obs_f + K @ (y - x_obs_f - (y - mu_obs_f))
    #         = A_obs_f + K @ (mu_obs_f - x_obs_f)
    #         = A_obs_f - K @ A_obs_f
    #         = (I - K) @ A_obs_f  ... but A_obs_f is (n_ens, n_obs), K is (n_obs, n_obs)
    # So: A_obs_a = A_obs_f @ (I - K_obs).T  ... let me check dimensions
    # A_obs_f: (n_ens, n_obs), K_obs: (n_obs, n_obs)
    # (I - K_obs).T: (n_obs, n_obs)
    # A_obs_f @ (I - K_obs).T: (n_ens, n_obs) ✓
    
    A_obs_a = A_obs_f @ (np.eye(n_obs) - K_obs.T)  # (n_ens, n_obs)
    
    # Build the analysis ensemble
    X_a = X_f.copy()
    
    # Set observed coordinates
    X_a[:, obs_idx_arr] = mu_obs_a[np.newaxis, :] + A_obs_a
    
    # For unobserved coordinates, use regression
    # For unobserved coord j:
    # C_of = Cov(x_j_f, x_obs_f) = A_f[:, j].T @ A_obs_f / (n_ens-1), shape (1, n_obs)
    # Cross-covariance: C_jo = A_j_f.T @ A_obs_f / (n_ens-1)  (1, n_obs)
    # Regression coefficient: beta_j = C_jo @ C_oo^{-1}  (1, n_obs)
    # 
    # The analysis correction for unobserved j:
    # delta_j = beta_j @ (x_obs_a - x_obs_f)
    #         = beta_j @ (mu_obs_a - mu_obs_f) + beta_j @ (A_obs_a - A_obs_f)
    # 
    # But we can also write this using the Kalman gain structure:
    # The full gain for unobserved j is: K_j = C_jo @ S_inv
    # delta_j = K_j @ innovation  for the mean
    # For anomalies: the correction involves the cross-covariance
    
    # More directly:
    # x_j_a = x_j_f + C_jo @ S_inv @ (y - x_obs_f)
    #       = x_j_f + C_jo @ S_inv @ innovation - C_jo @ S_inv @ A_obs_f
    #       = x_j_f + K_j @ innovation - K_j @ A_obs_f
    # where K_j = C_jo @ S_inv (shape 1, n_obs)
    
    # Precompute C_oo_inv for efficiency
    try:
        C_oo_inv = np.linalg.inv(C_oo_reg)
    except np.linalg.LinAlgError:
        C_oo_inv = np.linalg.pinv(C_oo_reg)
    
    # For each unobserved coordinate, compute the correction
    for j in range(n_state):
        if j in obs_idx_set:
            continue
        
        # Anomaly of unobserved coord j in forecast
        A_j_f = X_f[:, j] - np.mean(X_f[:, j])  # (n_ens,)
        
        # Cross-covariance: C_jo = A_j_f.T @ A_obs_f / (n_ens-1), shape (1, n_obs)
        C_jo = (A_j_f[np.newaxis, :] @ A_obs_f) / (n_ens - 1)  # (1, n_obs)
        
        # Gain for unobserved j: K_j = C_jo @ S_inv, shape (1, n_obs)
        K_j = C_jo @ S_inv  # (1, n_obs)
        
        # Mean correction: K_j @ innovation
        delta_mu_j = (K_j @ innovation)[0]
        
        # Anomaly correction: -K_j @ A_obs_f, shape (n_ens,)
        delta_A_j = -(A_obs_f @ K_j.T)[:, 0]  # (n_ens,)
        
        # Apply update
        X_a[:, j] += delta_mu_j + delta_A_j
    
    # Clip to valid range
    X_a = np.clip(X_a, -100.0, 100.0)
    
    # Ensure finite values
    X_a = np.where(np.isfinite(X_a), X_a, 0.0)
    
    return X_a
# EVOLVE-BLOCK-END
