"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
# Global deadline tracking
_start_time = time.time()
_DEADLINE = _start_time + 145.0  # Safety margin before 150s kill

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with adaptive inflation and localization.
    """
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
    # Using unbiased estimate: divide by M-1
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
    # Apply exponential localization to P_xo to reduce spurious correlations
    N_all = np.arange(N)
    N_obs_arr = np.array(obs_idx)
    
    # Compute circular distance
    dist = np.abs(N_all[:, np.newaxis] - N_obs_arr[np.newaxis, :])
    dist = np.minimum(dist, N - dist)  # Circular distance
    
    # Exponential localization: exp(-0.5 * (dist / L)^2)
    # L=2.0 better captures periodic correlations in 20-var Lorenz-96
    L = 2.0
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)
    
    # Localize P_xo
    P_xo_loc = P_xo * loc_matrix
    
    # S = P_oo + R with small Tikhonov regularization for stability
    S = P_oo + R + 1e-8 * np.eye(N_obs)
    
    # Compute inverse of S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    # K = P_xo_loc @ S_inv  (N x N_obs)
    K = P_xo_loc @ S_inv
    
    # 7. Update Mean
    mean_a = mean_b + K @ d
    
    # 8. Update Perturbations
    # W_a_T = (W_b.T - K @ H_W_b_T)
    H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    update_part = K @ H_W_b_T  # N x M
    W_a_T = W_b.T - update_part
    
    # 9. Adaptive Inflation (smoother norm-based approach)
    # Compare squared innovation norm to expected squared norm
    # Expected: E[||d||^2] = trace(S) = sum of diagonal of S
    expected_norm_sq = np.trace(S)
    actual_norm_sq = float(d @ d)
    
    # Ratio of expected to actual: if actual < expected, ensemble too confident
    if actual_norm_sq > 1e-10:
        ratio = expected_norm_sq / actual_norm_sq
    else:
        ratio = 1.0
    
    # Apply a damped inflation: use sqrt of ratio but with soft clipping
    # This is smoother than hard clipping
    raw_inflation = np.sqrt(ratio)
    
    # Soft-clip: blend between 1.0 and raw_inflation using a tanh-like smooth function
    # If ratio is close to 1, inflation ~ 1.0; if ratio >> 1, inflation approaches cap
    cap = 1.08
    # Smooth saturation: inflation = 1 + (cap - 1) * tanh((raw_inflation - 1) / 0.05)
    if raw_inflation > 1.0:
        inflation_factor = 1.0 + (cap - 1.0) * np.tanh((raw_inflation - 1.0) / 0.05)
    else:
        inflation_factor = 1.0
    
    # Apply inflation to perturbations
    W_a_T *= inflation_factor
    
    # Re-center the perturbations to ensure the mean remains mean_a
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]
    
    # 10. Construct Analysis Ensemble
    X_a = mean_a + W_a_T.T
    
    # 11. Safety checks: ensure finite and within bounds
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
