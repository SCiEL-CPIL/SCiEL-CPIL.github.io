import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Smoother (LES) with adaptive inflation.
    Updates each state variable using only local observations to avoid spurious correlations.
    """
    n_members = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Convert to numpy arrays if needed
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_obs = len(observed_indices)
    
    # 1. Compute global ensemble mean
    ensemble_mean = np.mean(ensemble, axis=0)
    
    # 2. Initialize updated ensemble
    updated_ensemble = np.zeros_like(ensemble)
    
    # Localization parameters
    # Window size for local observations
    window_size = 5 
    # Inflation parameters
    base_inflation = 1.02
    
    # 3. Loop over each state variable
    for i in range(n_state):
        # Determine local observation indices
        # In L96, indices are periodic (0-19).
        # We select observed indices j such that the circular distance |i-j| <= window_size
        local_obs_indices = []
        for j in observed_indices:
            # Calculate circular distance
            dist = abs(i - j)
            dist = min(dist, n_state - dist)
            if dist <= window_size:
                local_obs_indices.append(j)
        
        # If no local observations, keep the background (or apply only inflation)
        if len(local_obs_indices) == 0:
            updated_ensemble[:, i] = ensemble[:, i]
            continue
            
        # Extract local data
        # Anomalies for state i
        a_i = ensemble[:, i] - ensemble_mean[i]
        
        # Anomalies for local observations
        # Note: observed_indices are indices into the state vector
        # We need the mean of the observed states
        local_obs_mean = ensemble_mean[local_obs_indices]
        A_obs = ensemble[:, local_obs_indices] - local_obs_mean
        
        # Dimensions:
        # a_i: (N,)
        # A_obs: (N, M) where M = len(local_obs_indices)
        
        # Compute Covariances
        # C_xy: Cov(x_i, y_j) -> (1, M)
        C_xy = (a_i[:, np.newaxis] * A_obs).mean(axis=0) / (n_members - 1)
        
        # C_yy: Cov(y_j, y_k) -> (M, M)
        C_yy_sample = (A_obs.T @ A_obs) / (n_members - 1)
        
        # Add observation noise
        C_yy = C_yy_sample + observation_variance * np.eye(len(local_obs_indices))
        
        # Compute Gain
        # K = C_xy @ inv(C_yy)
        try:
            # C_xy is (1, M), C_yy is (M, M)
            # We want K (1, M)
            # K = C_xy @ inv(C_yy)
            # Using solve for stability: K.T = inv(C_yy).T @ C_xy.T
            # C_yy is symmetric, so inv(C_yy).T = inv(C_yy)
            K = np.linalg.solve(C_yy.T, C_xy.T).T
        except np.linalg.LinAlgError:
            K = C_xy @ np.linalg.pinv(C_yy)
            
        # Innovation
        # y is the observation vector, local_obs_mean is the background mean of obs
        y_local = observations[np.searchsorted(observed_indices, local_obs_indices)]
        innovation = y_local - local_obs_mean
        
        # Update Mean
        updated_mean_i = ensemble_mean[i] + K @ innovation
        
        # Update Anomalies
        # a^a_i = a_i - K @ A_obs
        # This is the standard EnKF anomaly update
        a_updated = a_i - K @ A_obs
        
        # Reconstruct state i
        updated_ensemble[:, i] = updated_mean_i + a_updated
        
    # 4. Adaptive Inflation
    # Calculate spread at observed locations
    # We need the updated mean to check spread
    updated_mean = np.mean(updated_ensemble, axis=0)
    
    # Check spread at observed indices
    obs_spread = np.std(updated_ensemble[:, observed_indices] - updated_mean[observed_indices], axis=0)
    mean_spread = np.mean(obs_spread)
    
    # Check innovation magnitude (using background mean for stability)
    bg_innovation = observations - ensemble_mean[observed_indices]
    innovation_std = np.std(bg_innovation)
    
    # Determine inflation factor
    # If spread is too small relative to innovation, inflate
    if mean_spread < 1e-6:
        inflation_factor = 1.2
    elif innovation_std > 2.0 * mean_spread:
        # Scale inflation based on ratio
        ratio = innovation_std / mean_spread
        # Cap inflation to avoid divergence
        inflation_factor = 1.0 + 0.1 * np.log(ratio) 
        inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    else:
        inflation_factor = base_inflation
        
    # Apply inflation around the updated mean
    updated_ensemble = updated_mean + inflation_factor * (updated_ensemble - updated_mean)
    
    # Ensure finiteness and bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
