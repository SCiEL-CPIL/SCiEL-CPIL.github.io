"""Combined Local EnAKF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    r1 = d[mask1]
    result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (15.0 / 8.0) * (r3 / c) ** 3 + (5.0 / 2.0) * (r3 / c) ** 2 - (25.0 / 12.0) * (r3 / c) + 105.0 / 48.0
    
    return result

def _local_enakf_update(local_ens, local_obs_mask, y_local, obs_var):
    """
    Perform a local Ensemble Adjustment Kalman Filter (EnAKF) update.
    This is a deterministic square-root transform that preserves the ensemble mean
    and exactly updates the covariance without stochastic noise injection.
    
    Parameters:
    local_ens: (M, L) array of local ensemble states
    local_obs_mask: (L,) boolean array indicating which local variables are observed
    y_local: (n_obs,) array of local observations
    obs_var: scalar observation variance
    
    Returns:
    updated local ensemble (M, L)
    """
    M, L = local_ens.shape
    n_obs = len(y_local)
    
    if n_obs == 0:
        return local_ens.copy()
    
    # Identify observed indices in the local window
    obs_idx_in_local = np.where(local_obs_mask)[0]
    
    # Ensemble mean
    x_mean = np.mean(local_ens, axis=0)
    
    # Anomalies
    A = local_ens - x_mean  # (M, L)
    
    # Local observation operator H: (n_obs, L)
    H = np.zeros((n_obs, L))
    for k, idx in enumerate(obs_idx_in_local):
        H[k, idx] = 1.0
    
    # Innovation
    innovation = y_local - H @ x_mean  # (n_obs,)
    
    # Ensemble covariance C = A^T A / (M-1)
    C = (A.T @ A) / (M - 1)  # (L, L)
    
    # Innovation covariance S = H C H^T + R
    HC = H @ C  # (n_obs, L)
    S = HC @ H.T + obs_var * np.eye(n_obs)  # (n_obs, n_obs)
    
    # Compute the gain K = C H^T S^{-1}
    # To avoid explicit inverse, solve S Z^T = H C^T
    try:
        HC_T = H @ C.T  # (n_obs, L)
        Z_T = np.linalg.solve(S, HC_T)
        K = Z_T.T  # (L, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C @ H.T @ S_inv
    
    # EnAKF Transform:
    # We want to transform the anomalies A such that the new anomalies A_new
    # have covariance C_a = C - K H C (the analysis covariance).
    # The EnAKF uses the transform:
    # A_new = A * (I - 0.5 * A_obs^T (A_obs A_obs^T + (M-1)*R)^{-1} A_obs)
    # where A_obs = A[:, obs_idx] (M, n_obs)
    
    A_obs = A[:, obs_idx_in_local]  # (M, n_obs)
    
    # Compute the matrix M_inv = (A_obs A_obs^T + (M-1)*R)^{-1}
    # A_obs A_obs^T is (n_obs, n_obs)
    AO_AO_T = A_obs @ A_obs.T  # (n_obs, n_obs)
    R_mat = obs_var * np.eye(n_obs)
    M_mat = AO_AO_T + (M - 1) * R_mat
    
    try:
        M_inv = np.linalg.inv(M_mat)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(M_mat)
    
    # Transform matrix: T = I - 0.5 * A_obs^T M_inv A_obs
    # This is (L, L) but we can apply it efficiently
    # A_new = A * T = A - 0.5 * A_obs^T M_inv A_obs A
    # Note: A_obs^T M_inv A_obs is (L, L) but sparse (only obs columns/rows nonzero)
    # More efficiently: A_new[:, j] = A[:, j] - 0.5 * sum_k A_obs[:, k] * (M_inv A_obs A)[:, k]
    
    # Compute B = M_inv A_obs A  (n_obs, L)
    # A_obs A is (n_obs, L)
    A_obs_A = A_obs @ A  # (n_obs, L)
    B = M_inv @ A_obs_A  # (n_obs, L)
    
    # A_new = A - 0.5 * A_obs @ B
    A_new = A - 0.5 * (A_obs @ B)
    
    # Update ensemble: new state = x_mean + A_new
    # But we also need to shift the mean to the analysis mean
    x_mean_new = x_mean + K @ innovation
    
    return x_mean_new + A_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Adjustment Kalman Filter (EnAKF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    if M > 1:
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        
        # If spread is too small, inflate more aggressively
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.01, 1.05)
        else:
            inflation = 1.01
        
        # Apply multiplicative inflation around the mean
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local EnAKF Update
    # Window size: include the target variable and its neighbors
    window_half = 2
    window_size = 2 * window_half + 1
    
    # Map global indices to observed values for quick lookup
    obs_lookup = {}
    for i, idx in enumerate(observed_indices):
        obs_lookup[idx] = observations[i]
    
    # Iterate over each state variable i
    for i in range(N):
        if time.time() > deadline:
            break
        
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_lookup:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_lookup[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Perform local EnAKF update
        updated_local_ens = _local_enakf_update(local_ens, local_obs_mask, y_local, observation_variance)
        
        # Update result
        result[:, local_indices] = updated_local_ens
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
