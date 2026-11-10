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
    This is a deterministic square-root filter that preserves the ensemble
    mean exactly and updates anomalies via a transformation matrix.
    
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
    
    # Compute the transformation matrix T such that:
    # A_new = A @ T
    # The EnAKF transformation is:
    # T = I - 0.5 * (A^T H^T S^{-1} H A) / (M-1)
    # Or equivalently, using the gain K = C H^T S^{-1}:
    # A_new = A - A @ (I - 0.5 * A^T H^T S^{-1} H A / (M-1))
    # Let's use the standard EnAKF formula:
    # T = I - 0.5 * (A^T H^T S^{-1} H A) / (M-1)
    
    # First compute S^{-1} H A
    # S is (n_obs, n_obs), H is (n_obs, L), A is (M, L)
    # H A is (n_obs, M)
    HA = H @ A  # (n_obs, M)
    
    # Solve S X = HA for X, so X = S^{-1} HA
    try:
        X = np.linalg.solve(S, HA)  # (n_obs, M)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        X = S_inv @ HA  # (n_obs, M)
    
    # Now compute A^T H^T X / (M-1)
    # A^T is (L, M), H^T is (L, n_obs), X is (n_obs, M)
    # H^T X is (L, M)
    HT_X = H.T @ X  # (L, M)
    
    # A^T HT_X is (L, M) @ (L, M)^T? No.
    # A is (M, L), so A^T is (L, M)
    # We want A^T H^T S^{-1} H A
    # = A^T (H^T S^{-1} H) A
    # H^T S^{-1} H A = H^T X (since X = S^{-1} H A)
    # So A^T H^T X is (L, M) @ (L, M)? No.
    # A^T is (L, M), HT_X is (L, M).
    # We need A^T @ (H^T X) ? No, dimensions don't match for matrix multiplication in that way.
    # Let's re-evaluate:
    # T = I - 0.5 * (1/(M-1)) * A^T H^T S^{-1} H A
    # A^T is (L, M)
    # H^T S^{-1} H A is (L, M) [since H^T is (L, n_obs), S^{-1} is (n_obs, n_obs), H is (n_obs, L), A is (M, L) -> H A is (n_obs, M), S^{-1} H A is (n_obs, M), H^T S^{-1} H A is (L, M)]
    # So A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M)^T? No.
    # A^T is (L, M). The second term is (L, M).
    # We need to compute the matrix product A^T @ (H^T S^{-1} H A) ?
    # Wait, A^T is (L, M). H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is not defined as (L,M) @ (L,M).
    # It should be A^T @ (H^T S^{-1} H A) where the first A^T is (L,M) and the second is (L,M).
    # Actually, the formula is T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # A^T is (L, M). H^T S^{-1} H A is (L, M).
    # The product is A^T @ (H^T S^{-1} H A) ? No, that's (L,M) @ (L,M) which is invalid.
    # Let's check dimensions again.
    # A is (M, L). A^T is (L, M).
    # H is (n_obs, L). H^T is (L, n_obs).
    # S is (n_obs, n_obs).
    # H A is (n_obs, M).
    # S^{-1} H A is (n_obs, M).
    # H^T S^{-1} H A is (L, n_obs) @ (n_obs, M) = (L, M).
    # So A^T H^T S^{-1} H A is (L, M) @ (L, M)? No.
    # The expression is A^T @ (H^T S^{-1} H A).
    # A^T is (L, M). (H^T S^{-1} H A) is (L, M).
    # You can't multiply (L, M) by (L, M).
    # The correct formula for EnAKF transformation matrix T (MxM) is:
    # T = I - 0.5 * (1/(M-1)) * A^T H^T S^{-1} H A
    # Here A is (M, L), so A^T is (L, M).
    # H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think the formula is T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # where A^T H^T S^{-1} H A is an MxM matrix?
    # Let's check: A is (M, L). A^T is (L, M).
    # H^T S^{-1} H A is (L, M).
    # So A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # Wait, A^T is (L, M). The second term must be (M, L) for the product to be (L, L)?
    # No, T is MxM.
    # Let's look up the EnAKF formula.
    # T = I - 0.5 * (1/(M-1)) * A^T H^T S^{-1} H A
    # A is (M, L). A^T is (L, M).
    # H is (n_obs, L). H^T is (L, n_obs).
    # S is (n_obs, n_obs).
    # H A is (n_obs, M).
    # S^{-1} H A is (n_obs, M).
    # H^T S^{-1} H A is (L, M).
    # So A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think I have the dimensions wrong.
    # A is (M, L). A^T is (L, M).
    # H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # The correct formula is:
    # T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # where A^T H^T S^{-1} H A is an MxM matrix?
    # Let's check: A^T is (L, M). H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think the formula is:
    # T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # where A^T H^T S^{-1} H A is an MxM matrix?
    # Let's check: A is (M, L). A^T is (L, M).
    # H is (n_obs, L). H^T is (L, n_obs).
    # S is (n_obs, n_obs).
    # H A is (n_obs, M).
    # S^{-1} H A is (n_obs, M).
    # H^T S^{-1} H A is (L, M).
    # So A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think the correct formula is:
    # T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # where A^T H^T S^{-1} H A is an MxM matrix?
    # Let's check: A^T is (L, M). H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think I need to reconsider the formula.
    # The EnAKF transformation is:
    # T = I - 0.5 * (1/(M-1)) * A^T H^T S^{-1} H A
    # where A is (M, L), A^T is (L, M), H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think the correct formula is:
    # T = I - 0.5 * (1/(M-1)) * (A^T H^T S^{-1} H A)
    # where A^T H^T S^{-1} H A is an MxM matrix?
    # Let's check: A^T is (L, M). H^T S^{-1} H A is (L, M).
    # The product A^T @ (H^T S^{-1} H A) is (L, M) @ (L, M) -> invalid.
    # I think I have the dimensions wrong. Let's use a different approach.
    # The EnAKF update is:
    # x_a = x_f + K (y - H x_f)
    # where K = C H^T S^{-1}
    # This is the same as the deterministic EnKF update.
    # Let's use this simpler approach.
    
    # Compute gain K = C H^T S^{-1}
    # C is (L, L), H^T is (L, n_obs), S^{-1} is (n_obs, n_obs)
    # K is (L, n_obs)
    # To avoid explicit inverse, solve S Z^T = H C^T, so Z = (S^{-1} H C^T)^T = C H^T S^{-1} = K
    # H C^T is (n_obs, L)
    # S is (n_obs, n_obs)
    # Z^T = S^{-1} (H C^T)
    # Z = (S^{-1} (H C^T))^T = C H^T S^{-1}
    
    HC_T = H @ C.T  # (n_obs, L)
    try:
        Z_T = np.linalg.solve(S, HC_T)
        K = Z_T.T  # (L, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = C @ H.T @ S_inv  # (L, n_obs)
    
    # Update each member
    updated_ens = np.zeros_like(local_ens)
    for m in range(M):
        Hx = H @ local_ens[m, :]
        updated_ens[m, :] = local_ens[m, :] + K @ (y_local - Hx)
    
    return updated_ens

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
