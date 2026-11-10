import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Apply adaptive inflation
    # Calculate spread at observed locations
    spread_obs = np.std(anomalies[:, observed_indices], axis=0)
    # Target spread is roughly sqrt(observation_variance)
    # If spread is too low, inflate.
    target_spread = np.sqrt(observation_variance)
    # Simple adaptive inflation: scale anomalies such that spread matches target
    # We only want to inflate if spread is too small
    current_spread_mean = np.mean(spread_obs)
    if current_spread_mean > 1e-6:
        inflation_factor = np.clip(target_spread / current_spread_mean, 1.0, 1.5)
    else:
        inflation_factor = 1.0
    
    anomalies *= inflation_factor
    
    # 4. Compute localized ensemble covariance
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply localization: Gaspari-Cohn
    # Precompute distances for efficiency? N=20 is small, loop is fine.
    def gaspari_cohn(d, L=3.0):
        if d <= L/2:
            return (-0.25 * (2*d/L - 5)**5 + 5*(2*d/L - 3)**5 + 5*(2*d/L - 1)**5 + 5) / 4
        elif d <= L:
            return (0.5 * (2*d/L - 7)**5 - 5*(2*d/L - 5)**5 + 10*(2*d/L - 3)**5 - 5*(2*d/L - 1)**5 + 5) / 2
        else:
            return 0.0
            
    # Vectorized localization application
    indices = np.arange(N)
    # Distance matrix on a ring
    dist_matrix = np.abs(indices[:, None] - indices[None, :])
    dist_matrix = np.minimum(dist_matrix, N - dist_matrix)
    
    # Compute Gaspari-Cohn for all pairs
    # This might be slow if done element-wise in Python, but N=20 is tiny.
    # Let's just use the loop for clarity and correctness, or vectorize if possible.
    # Vectorized GC:
    d = dist_matrix
    L = 3.0
    gc = np.zeros_like(d)
    mask1 = d <= L/2
    mask2 = (d > L/2) & (d <= L)
    
    # Formula for mask1
    term1 = -0.25 * (2*d[mask1]/L - 5)**5 + 5*(2*d[mask1]/L - 3)**5 + 5*(2*d[mask1]/L - 1)**5 + 5
    gc[mask1] = term1 / 4
    
    # Formula for mask2
    term2 = 0.5 * (2*d[mask2]/L - 7)**5 - 5*(2*d[mask2]/L - 5)**5 + 10*(2*d[mask2]/L - 3)**5 - 5*(2*d[mask2]/L - 1)**5 + 5
    gc[mask2] = term2 / 2
    
    P_f *= gc

    # 5. Observation operator H is implicit via indices
    H_P_H = P_f[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 6. Kalman Gain K_g = P_f H^T S^-1
    P_f_obs = P_f[:, observed_indices]
    
    try:
        S_inv_P_f_obs_T = np.linalg.solve(S, P_f_obs.T)
        K_g = S_inv_P_f_obs_T.T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K_g = P_f_obs @ S_inv

    # 7. Innovation
    obs_mean = mean_f[observed_indices]
    innovation = obs - obs_mean
    
    # 8. Update mean
    mean_a = mean_f + K_g @ innovation
    
    # 9. Update anomalies
    # Construct H^T K_g^T
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
