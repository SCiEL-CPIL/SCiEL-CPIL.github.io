import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a Localized Ensemble Kalman Filter (EnKF)
    with multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of the observed state variables.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        The updated ensemble after assimilation.
    """
    # Ensure numpy arrays
    ensemble = np.array(ensemble, dtype=float)
    observations = np.array(observations, dtype=float)
    observed_indices = np.array(observed_indices, dtype=int)
    
    num_members, num_states = ensemble.shape
    num_obs = len(observations)
    
    # 1. Compute mean and perturbations
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    
    # 2. Multiplicative Inflation
    # Increase spread to handle under-dispersion in chaotic systems
    inflation_factor = 1.08
    perturbations *= inflation_factor
    
    # 3. Observation Space Perturbations
    # perturbations_obs has shape (num_members, num_obs)
    perturbations_obs = perturbations[:, observed_indices]
    
    # 4. Covariances
    # Observation space covariance S_obs: (num_obs, num_obs)
    # S_obs = 1/(N-1) * E_obs * E_obs.T + R
    E_obs_T = perturbations_obs.T
    S_obs = (perturbations_obs @ E_obs_T) / (num_members - 1)
    S_obs += np.eye(num_obs) * observation_variance
    
    # Cross-covariance P_x_obs: (num_states, num_obs)
    # P_x_obs = 1/(N-1) * E * E_obs.T
    P_x_obs = (perturbations @ E_obs_T) / (num_members - 1)
    
    # 5. Kalman Gain
    # K = P_x_obs * S_obs^-1
    # Use solve for numerical stability
    try:
        # K shape: (num_states, num_obs)
        K = np.linalg.solve(S_obs.T, P_x_obs.T).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        K = P_x_obs @ np.linalg.pinv(S_obs)
        
    # 6. Localization
    # Apply a Gaussian localization function to the gain K
    # Distance between state i and observed state j
    loc_scale = 2.0  # Tighter localization as per findings
    sigma2 = 2.0 * (loc_scale ** 2)
    
    # K is (num_states, num_obs)
    # observed_indices maps obs index j to state index idx_j
    # We need to compute distance between state i (row of K) and state idx_j (col of K)
    
    # Vectorized localization
    # i_indices: (num_states, 1)
    # j_indices: (1, num_obs) -> values are observed_indices
    i_indices = np.arange(num_states).reshape(-1, 1)
    j_indices = observed_indices.reshape(1, -1)
    
    # Periodic distance in Lorenz-96
    diff = np.abs(i_indices - j_indices)
    diff = np.minimum(diff, num_states - diff)
    
    # Gaussian kernel
    loc_weights = np.exp(-(diff ** 2) / sigma2)
    
    # Apply localization to K
    K *= loc_weights
    
    # 7. Update Mean
    # Innovation: y - H(mean)
    H_mean = mean[observed_indices]
    innovation = observations - H_mean
    
    mean_updated = mean + K @ innovation
    
    # 8. Update Perturbations
    # e_i^a = e_i + K * (H e_i)
    # H e_i is the row of perturbations_obs
    # perturbations_updated = perturbations + K @ perturbations_obs.T ?
    # Let's check shapes:
    # perturbations: (N, M)
    # K: (M, M_obs)
    # perturbations_obs: (N, M_obs)
    # We want to add to each member i: K @ perturbations_obs[i]
    # This is equivalent to: perturbations + (K @ perturbations_obs.T).T
    # K @ perturbations_obs.T is (M, N)
    # .T is (N, M)
    
    perturbations_updated = perturbations + (K @ perturbations_obs.T).T
    
    # 9. Reconstruct Ensemble
    updated_ensemble = mean_updated + perturbations_updated
    
    # 10. Safety Checks
    # Ensure finite values
    if not np.all(np.isfinite(updated_ensemble)):
        # If anything goes wrong, return the original ensemble or a clipped version
        # Returning original is safer than NaNs
        return ensemble.copy()
        
    # Clip to [-100, 100] as per contract
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
