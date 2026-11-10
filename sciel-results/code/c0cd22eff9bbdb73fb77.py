import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) update with tuned localization and adaptive inflation.
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Compute ensemble mean
    mean = np.mean(ensemble, axis=0)
    
    # Compute anomaly matrix (M, N)
    anomalies = ensemble - mean
    
    # Covariance: (N, N)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Localization: Smooth exponential with periodic distance
    # Tuned scale from 4.5 to 4.0 based on higher-scoring inspiration
    localization_scale = 4.0
    indices = np.arange(N)
    dist = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    dist = np.minimum(dist, N - dist)
    
    # Exponential localization
    loc_factor = np.exp(-dist / localization_scale)
    
    # Localize the covariance
    cov_loc = cov * loc_factor
    
    # Observation operator: H is (K, N)
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance R
    R = observation_variance * np.eye(K)
    
    # Innovation: y - H x
    # Handle case where observations might be a single vector (K,) or perturbed (M, K)
    if observations.ndim == 1:
        y_vec = observations
        y_mean = y_vec
        # For perturbed obs update, we need to expand y_vec to (M, K)
        obs_matrix = np.tile(y_vec, (M, 1))
    else:
        y_mean = np.mean(observations, axis=0)
        obs_matrix = observations
    
    # Innovation vector for mean update: (K,)
    innovation_mean = y_mean - H @ mean
    
    # Innovation covariance: S = H C H^T + R
    S = H @ cov_loc @ H.T + R
    
    # Gain: K_gain = C H^T S^-1
    try:
        # Solve S * x = innovation for stability
        S_inv_innov = np.linalg.solve(S, innovation_mean)
        # We still need S_inv for the gain matrix if we want to compute it explicitly,
        # but we can compute gain @ innovation directly.
        # gain = cov_loc @ H.T @ S_inv
        # gain @ innovation_mean = cov_loc @ H.T @ S_inv @ innovation_mean
        # S_inv @ innovation_mean is S_inv_innov
        update_vector = cov_loc @ H.T @ S_inv_innov
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        S_inv_innov = S_inv @ innovation_mean
        update_vector = cov_loc @ H.T @ S_inv_innov
    
    # Update mean
    mean_a = mean + update_vector
    
    # Update ensemble members using perturbed observations
    # innovations_i = y_i - H x_i
    innovations = obs_matrix - ensemble[:, observed_indices]  # (M, K)
    
    # Update each member: x_a_i = x_f_i + gain @ innovation_i
    # We need gain @ innovation_i for each i.
    # gain = cov_loc @ H.T @ S_inv
    # Let's compute S_inv explicitly for the member updates if not done above, 
    # or use the same S_inv_innov logic? No, innovations differ.
    # We need S_inv.
    if 'S_inv' not in locals():
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    gain = cov_loc @ H.T @ S_inv  # (N, K)
    
    # result = ensemble + innovations @ gain.T
    result = ensemble + innovations @ gain.T
    
    # Adaptive Inflation
    # Compute ensemble variance at observed points
    ens_var_obs = np.var(ensemble[:, observed_indices], axis=0)  # (K,)
    obs_var = observation_variance
    
    # Compute spread ratio
    spread_ratio = np.mean(ens_var_obs) / obs_var if obs_var > 0 else 1.0
    
    # Smooth adaptive inflation
    # Target ratio slightly lower than before to allow more inflation if needed
    target_ratio = 1.15
    deviation = (spread_ratio - target_ratio) / target_ratio
    
    # Increased beta for stronger correction
    beta = 0.20
    inflation = 1.0 + beta * np.tanh(-deviation)
    
    # Clamp to reasonable bounds
    inflation = np.clip(inflation, 0.95, 1.35)
    
    # Apply inflation around the updated mean
    result = mean_a + inflation * (result - mean_a)
    
    # Ensure the result is finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110:
        return result
    
    return result
