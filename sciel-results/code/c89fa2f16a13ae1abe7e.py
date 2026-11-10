"""Combined EnKF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Precompute localization matrix for N=20 (ring topology)
# Gaspari-Cohn function with correlation length c
def _gc(d, c):
    """Gaspari-Cohn correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    # d <= c/2
    mask1 = d <= 0.5 * c
    r1 = d[mask1]
    result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    # c/2 < d <= 3c/2
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    r2 = d[mask2]
    result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    # 3c/2 < d <= 5c/2
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    r3 = d[mask3]
    result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (15.0 / 8.0) * (r3 / c) ** 3 + (5.0 / 2.0) * (r3 / c) ** 2 - (25.0 / 12.0) * (r3 / c) + 1.0 / 6.0
    # d > 5c/2: result remains 0
    return result

# Precompute localization matrix for N=20
_N = 20
_C = 2.0  # correlation length in indices
_idx = np.arange(_N)
_dmat = np.abs(_idx[:, None] - _idx[None, :])
_dmat = np.minimum(_dmat, _N - _dmat)
_LOC = _gc(_dmat, _C)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with adaptive inflation and localization
    for Lorenz-96 partial-observation assimilation.
    """
    t0 = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten().astype(float)
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Adaptive inflation based on ensemble spread vs observation mismatch
    # Use a simple adaptive scheme: inflate if spread is too small relative to innovation
    obs_mean_f = mean_f[observed_indices]
    innovation = obs - obs_mean_f
    
    # Estimate ensemble spread at observed points
    spread_obs = np.std(ensemble[:, observed_indices], axis=0)
    # Target spread should be comparable to observation noise
    target_spread = np.sqrt(observation_variance)
    
    # Adaptive inflation factor
    ratio = np.mean(spread_obs) / (target_spread + 1e-10)
    if ratio < 0.8:
        inflation_factor = 1.0 + 0.05 * (1.0 - ratio)
    elif ratio > 1.5:
        inflation_factor = 1.0 - 0.02 * (ratio - 1.5)
    else:
        inflation_factor = 1.0
    
    # Clamp inflation to reasonable range
    inflation_factor = np.clip(inflation_factor, 0.98, 1.15)
    
    # Apply inflation to anomalies
    anomalies *= inflation_factor
    
    # 4. Compute localized ensemble covariance
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply precomputed localization
    P_f_loc = P_f * _LOC
    
    # 5. Observation operator H is implicit via indices
    # S = H P_f_loc H^T + R
    H_P_H = P_f_loc[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 6. Kalman Gain K_g = P_f_loc H^T S^-1
    P_f_obs = P_f_loc[:, observed_indices]
    
    try:
        S_inv_P_f_obs_T = np.linalg.solve(S, P_f_obs.T)
        K_g = S_inv_P_f_obs_T.T  # Shape: N x K
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_g = P_f_obs @ S_inv  # Shape: N x K
    
    # 7. Innovation (already computed)
    # 8. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 9. Update anomalies using ensemble Kalman filter perturbation method
    # M_update = I - H^T K_g^T
    H_T_K_g_T = np.zeros((N, N))
    for k, idx in enumerate(observed_indices):
        H_T_K_g_T[idx, :] = K_g[:, k]
        
    M_update = np.eye(N) - H_T_K_g_T
    anomalies_a = anomalies @ M_update
    
    # 10. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Safety: ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 100:  # 100s safety margin before 120s limit
        pass  # Already returned
    
    return ensemble_a

# EVOLVE-BLOCK-START
# (The analyze function above is the main entry point)
# EVOLVE-BLOCK-END
