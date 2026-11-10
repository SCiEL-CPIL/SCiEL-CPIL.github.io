"""Combined EnKF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with inflation and localization.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (K,) or (1, K) array of noisy observations for observed indices.
    observed_indices: array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Apply static inflation to prevent filter collapse
    inflation_factor = 1.08
    anomalies *= inflation_factor
    
    # 4. Compute ensemble covariance with localization
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply Gaspari-Cohn localization to avoid spurious correlations
    # Precompute localization matrix for efficiency (N=20 is small)
    L = 4.0  # Correlation length scale in indices
    
    def gaspari_cohn(d, L):
        d = np.abs(d)
        d = np.minimum(d, N - d)  # Ring distance
        result = np.zeros_like(d, dtype=float)
        # First region: d <= L/2
        mask1 = d <= L/2
        x = 2*d[mask1]/L - 5
        result[mask1] = (-0.25*x**5 + 5*(2*d[mask1]/L - 3)**5 + 5*(2*d[mask1]/L - 1)**5 + 5) / 4
        
        # Second region: L/2 < d <= L
        mask2 = (d > L/2) & (d <= L)
        x = 2*d[mask2]/L - 7
        result[mask2] = (0.5*x**5 - 5*(2*d[mask2]/L - 5)**5 + 10*(2*d[mask2]/L - 3)**5 - 5*(2*d[mask2]/L - 1)**5 + 5) / 2
        
        # d > L: result remains 0
        return result
    
    # Build localization matrix
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            if d <= L:
                if d <= L/2:
                    x1 = 2*d/L - 5
                    x2 = 2*d/L - 3
                    x3 = 2*d/L - 1
                    loc_matrix[i, j] = (-0.25*x1**5 + 5*x2**5 + 5*x3**5 + 5) / 4
                else:
                    x1 = 2*d/L - 7
                    x2 = 2*d/L - 5
                    x3 = 2*d/L - 3
                    x4 = 2*d/L - 1
                    loc_matrix[i, j] = (0.5*x1**5 - 5*x2**5 + 10*x3**5 - 5*x4**5 + 5) / 2
            else:
                loc_matrix[i, j] = 0.0
    
    P_f_loc = P_f * loc_matrix
    
    # 5. Observation operator H is implicit via indices
    # S = H P_f H^T + R
    H_P_H = P_f_loc[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 6. Kalman Gain K_g = P_f H^T S^-1
    P_f_obs = P_f_loc[:, observed_indices]
    
    try:
        S_inv_P_f_obs_T = np.linalg.solve(S, P_f_obs.T)
        K_g = S_inv_P_f_obs_T.T  # Shape: N x K
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_g = P_f_obs @ S_inv  # Shape: N x K
    
    # 7. Innovation
    obs_mean = mean_f[observed_indices]
    innovation = obs - obs_mean
    
    # 8. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 9. Update anomalies
    # Construct H^T K_g^T matrix (N x N)
    H_T_K_g_T = np.zeros((N, N))
    for k, idx in enumerate(observed_indices):
        H_T_K_g_T[idx, :] = K_g[:, k]
        
    M_update = np.eye(N) - H_T_K_g_T
    anomalies_a = anomalies @ M_update
    
    # 10. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    return ensemble_a
# EVOLVE-BLOCK-END
