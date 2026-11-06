"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
# Global deadline tracking
_start_time = time.time()
_DEADLINE = _start_time + 140.0  # Safety margin before 150s kill

# State for adaptive inflation (persistent across calls)
_inflation_history = []

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with adaptive inflation, localization,
    and ensemble adjustment.
    """
    global _inflation_history
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)
    
    # Ensure float64 for precision
    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    
    M, N = X_b.shape  # M=16, N=20
    N_obs = len(obs_idx)
    
    # 1. Compute background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N
    
    # 2. Compute background covariance P_b (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # 3. Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    
    # 4. Observation error covariance R (N_obs x N_obs)
    R = np.eye(N_obs) * R_val
    
    # 5. Innovation (mean difference)
    H_mean = mean_b[obs_idx]
    d = y - H_mean
    
    # 6. Compute Kalman Gain with localization
    N_all = np.arange(N)
    N_obs_arr = np.array(obs_idx)
    
    # Compute circular distance
    dist = np.abs(N_all[:, np.newaxis] - N_obs_arr[np.newaxis, :])
    dist = np.minimum(dist, N - dist)  # Circular distance
    
    # Cosine localization with adaptive length scale
    # Tighter length scale to reduce spurious correlations
    L = 1.0
    # Cosine taper: smooth at boundaries, zero beyond L
    loc_matrix = np.where(dist <= L, 
                          0.5 * (1 + np.cos(np.pi * dist / L)), 
                          0.0)
    
    # Localize P_xo
    P_xo_loc = P_xo * loc_matrix
    
    # S = P_oo + R
    S = P_oo + R
    
    # Compute inverse of S using solve for numerical stability
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # K = P_xo_loc @ S_inv  (N x N_obs)
    K = P_xo_loc @ S_inv
    
    # 7. Update Mean
    mean_a = mean_b + K @ d
    
    # 8. Update Perturbations using ensemble adjustment (EnKF-A)
    # This preserves the ensemble mean and is more stable
    # H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    # Instead, use the ensemble adjustment formula:
    # W_a = W_b @ (I - 0.5 * H^T @ S^{-1} @ H)
    
    H_W_b = W_b[:, obs_idx]  # M x N_obs
    # S^{-1} @ H_W_b: N_obs x M
    S_inv_H_W_b = S_inv @ H_W_b
    # H^T @ S^{-1} @ H: M x M
    Ht_Sinv_H = H_W_b.T @ S_inv_H_W_b
    # Update matrix: I - 0.5 * Ht_Sinv_H
    update_mat = np.eye(M) - 0.5 * Ht_Sinv_H
    # W_a = W_b @ update_mat
    W_a = W_b @ update_mat
    
    # 9. Adaptive Inflation
    # Track innovation statistics over recent updates
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    
    ratio = expected_innov_var / actual_innov_var
    _inflation_history.append(ratio)
    
    # Use exponential moving average for stability
    if len(_inflation_history) > 10:
        _inflation_history = _inflation_history[-10:]
    
    if len(_inflation_history) >= 3:
        avg_ratio = np.mean(_inflation_history)
    else:
        avg_ratio = ratio
    
    # Inflation factor based on average ratio
    inflation_factor = np.sqrt(max(avg_ratio, 1.0))
    
    # Cap inflation to prevent divergence
    inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    
    # Apply inflation to perturbations
    W_a *= inflation_factor
    
    # Re-center the perturbations
    mean_W_a = np.mean(W_a, axis=0)
    W_a -= mean_W_a
    
    # 10. Construct Analysis Ensemble
    X_a = mean_a + W_a
    
    # 11. Safety checks
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
