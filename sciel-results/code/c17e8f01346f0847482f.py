"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Smoother (LEST) with multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Normalize observations to a single vector of length n_obs
    if observations.ndim == 1:
        y = observations
    else:
        # If 2D, assume it's (n_members, n_obs) and take the mean or first row?
        # In standard DA, obs are from truth, so they are identical for all members.
        # If they differ, taking the mean is a safe bet for a single "truth" estimate.
        y = np.mean(observations, axis=0)
        
    # Initialize updated ensemble
    updated_ensemble = ensemble.copy()
    
    # Local window size (number of variables to consider around each obs)
    # Must be odd to have a center. 7 means 3 left, 3 right.
    local_window = 7
    half_window = local_window // 2
    
    # Loop over each observation
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        obs_val = y[j]
        
        # Determine local indices (periodic)
        local_indices = []
        for k in range(-half_window, half_window + 1):
            local_indices.append((obs_idx + k) % n_vars)
            
        # Extract local ensemble states: shape (n_members, local_window)
        x_local = updated_ensemble[:, local_indices]
        
        # Compute local mean and covariance
        x_local_mean = np.mean(x_local, axis=0)
        x_local_anom = x_local - x_local_mean
        # Covariance P_local: shape (local_window, local_window)
        P_local = (x_local_anom.T @ x_local_anom) / (n_members - 1)
        
        # Observation operator H_local: shape (1, local_window)
        # The observation corresponds to the center of the window (index half_window in local_indices)
        # Since local_indices is constructed as obs_idx - half ... obs_idx + half,
        # the center is at index half_window.
        H_local = np.zeros(local_window)
        H_local[half_window] = 1.0
        
        # Compute Kalman Gain for this local observation
        # S = H P H^T + R (scalar)
        HP = P_local @ H_local  # shape (local_window,)
        S = H_local @ HP + observation_variance  # scalar
        
        # Avoid division by zero
        if S < 1e-12:
            K_local = np.zeros(local_window)
        else:
            K_local = HP / S  # shape (local_window,)
            
        # Innovation for each member
        # y - H x_i for each member i
        # H x_i is just the element at index half_window in x_local[i]
        Hx_local = x_local[:, half_window]  # shape (n_members,)
        innovations = obs_val - Hx_local  # shape (n_members,)
        
        # Update local ensemble members
        # x_i_new = x_i + K_local * innovation_i
        # K_local is (local_window,), innovations is (n_members,)
        # We need to add K_local * innovations[i] to each row of x_local[i]
        update_matrix = np.outer(innovations, K_local)  # shape (n_members, local_window)
        x_local_updated = x_local + update_matrix
        
        # Write back to updated_ensemble
        updated_ensemble[:, local_indices] = x_local_updated
        
    # Multiplicative Inflation
    # Inflate the ensemble around the mean to maintain spread
    inflation_factor = 1.03
    mean_ens = np.mean(updated_ensemble, axis=0)
    updated_ensemble = inflation_factor * (updated_ensemble - mean_ens) + mean_ens
    
    # Ensure finite values and clip
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
