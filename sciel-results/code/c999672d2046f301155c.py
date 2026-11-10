"""Local ETKF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _local_etkf_update(ensemble_window, obs_window, obs_var, inflation=1.05):
    """
    Perform local ETKF update on a window of variables.
    
    Parameters:
    ensemble_window: (M, W) array of ensemble states for the window.
    obs_window: (K,) array of observations for the window.
    obs_var: scalar observation variance.
    inflation: inflation factor.
    
    Returns:
    updated_ensemble_window: (M, W) array of updated ensemble states.
    """
    M, W = ensemble_window.shape
    K = len(obs_window)
    
    if K == 0:
        return ensemble_window
    
    # Compute mean and anomalies
    mean_f = np.mean(ensemble_window, axis=0)
    anomalies = ensemble_window - mean_f
    
    # Apply inflation
    anomalies = anomalies * inflation
    
    # Compute local ensemble covariance
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Regularize
    P_f += np.eye(W) * 1e-6
    
    # Observation operator: all variables in the window are observed
    # H = I_W, so H P_f H^T = P_f
    S = P_f + np.eye(W) * obs_var
    
    # Kalman gain
    try:
        L_chol = np.linalg.cholesky(S)
        X = np.linalg.solve(L_chol, P_f.T)
        K_g = np.linalg.solve(L_chol.T, X).T  # Shape: W x W
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
            K_g = P_f @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K_g = P_f @ S_inv
    
    # Innovation
    innovation = obs_window - mean_f
    mean_a = mean_f + K_g @ innovation
    
    # Update anomalies
    M_update = np.eye(W) - K_g
    anomalies_a = anomalies @ M_update.T
    
    # Reconstruct
    ensemble_a = mean_a[None, :] + anomalies_a
    
    return ensemble_a


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (ETKF) with sliding windows.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (K,) or (1, K) array of noisy observations for observed indices.
    observed_indices: array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten().astype(float)
    obs_idx = np.asarray(observed_indices).astype(int)
    
    # Adaptive inflation
    mean_f = np.mean(ensemble, axis=0)
    spread_obs = np.std(ensemble[:, obs_idx], axis=0)
    innovation = obs - mean_f[obs_idx]
    expected_innov = np.sqrt(spread_obs**2 + observation_variance)
    ratio = np.mean(np.abs(innovation)) / (np.mean(expected_innov) + 1e-12)
    
    base_inflation = 1.03
    adaptive_inflation = np.clip(0.04 * (ratio - 1.0), 0.0, 0.08)
    inflation = base_inflation + adaptive_inflation
    inflation = np.clip(inflation, 1.0, 1.15)
    
    # Sliding window approach
    # Window size: use a window that covers the observed index and neighbors
    window_half = 4  # 4 on each side, so window size 9
    
    # Initialize result
    result = ensemble.copy()
    
    # Process each observed index with its local window
    for i, idx in enumerate(obs_idx):
        # Define window
        start = (idx - window_half) % N
        # Get window indices with wrap-around
        window_indices = [(idx - window_half + j) % N for j in range(2 * window_half + 1)]
        window_size = len(window_indices)
        
        # Extract ensemble window
        ens_window = ensemble[:, window_indices]
        
        # Extract observations for this window (only the center is observed)
        # For local ETKF, we treat all variables in the window as potentially observed
        # But we only have observation for the center
        # Create a full observation vector with NaN for unobserved, then handle
        
        # Simpler approach: only update using the observed variable
        # Use a reduced observation operator
        
        # For now, use a simple local update with just the observed variable
        # Extract the observed variable from the window
        obs_local = np.array([obs[i]])
        
        # The observation operator H selects the center variable
        center_pos = window_half  # Position of observed variable in window
        
        # Perform local ETKF with reduced observation
        M_w, W = ens_window.shape
        
        mean_w = np.mean(ens_window, axis=0)
        anomalies_w = ens_window - mean_w
        anomalies_w = anomalies_w * inflation
        
        P_w = (anomalies_w.T @ anomalies_w) / (M_w - 1)
        P_w += np.eye(W) * 1e-6
        
        # H is a 1xW matrix selecting the center
        H = np.zeros((1, W))
        H[0, center_pos] = 1.0
        
        # S = H P H^T + R
        H_P_H = H @ P_w @ H.T
        S = H_P_H + np.array([[observation_variance]])
        
        # K = P H^T S^-1
        P_H_T = P_w @ H.T  # W x 1
        K_g = (P_H_T / S[0, 0]).reshape(W, 1)
        
        # Innovation
        innov = obs_local - mean_w[center_pos]
        mean_a = mean_w + (K_g @ innov.reshape(1, 1)).flatten()
        
        # Update anomalies
        M_update = np.eye(W) - (K_g @ H)
        anomalies_a = anomalies_w @ M_update.T
        
        # Reconstruct
        ens_a = mean_a[None, :] + anomalies_a
        
        # Update result
        for j, wi in enumerate(window_indices):
            result[:, wi] = ens_a[:, j]
        
        # Time guard
        if time.time() - t0 > 120:
            break
    
    # Ensure finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
