import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and inflation for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of the observed variables.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after the analysis step.
    """
    N_members, N_states = ensemble.shape
    N_obs = len(observed_indices)
    
    # 1. Compute mean and perturbations
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    
    # 2. Inflation (multiplicative)
    # A fixed inflation factor helps prevent ensemble collapse.
    # 1.02 is a conservative choice for chaotic systems.
    inflation_factor = 1.02
    perturbations *= inflation_factor
    
    # 3. Localization parameters
    # Radius of influence for cosine taper
    R_influence = 5.0
    
    def localization_weight(i, j):
        # Periodic distance on a ring of N_states
        dist = abs(i - j)
        dist = min(dist, N_states - dist)
        if dist < R_influence:
            return 0.5 * (1.0 + np.cos(np.pi * dist / R_influence))
        return 0.0

    # 4. Compute Covariances
    # P_xy: Covariance between all states (N_states) and observed states (N_obs)
    # Shape: (N_states, N_obs)
    # P_xy[i, k] = Cov(state_i, obs_k)
    
    # Extract observed perturbations
    # perturbations_obs shape: (N_members, N_obs)
    perturbations_obs = perturbations[:, observed_indices]
    
    # Compute raw covariances
    # Cov = X' X'_obs^T / (N-1)
    # X' is (N_members, N_states), X'_obs is (N_members, N_obs)
    # X' X'_obs^T is (N_states, N_obs)
    P_xy_raw = perturbations.T @ perturbations_obs
    P_xy_raw /= (N_members - 1)
    
    # Apply localization to P_xy
    P_xy = np.zeros((N_states, N_obs))
    for k, obs_idx in enumerate(observed_indices):
        for i in range(N_states):
            w = localization_weight(i, obs_idx)
            P_xy[i, k] = P_xy_raw[i, k] * w
            
    # P_yy: Covariance between observed states (N_obs, N_obs)
    # P_yy[k, m] = Cov(obs_k, obs_m)
    P_yy_raw = perturbations_obs.T @ perturbations_obs
    P_yy_raw /= (N_members - 1)
    
    # Apply localization to P_yy
    P_yy = np.zeros((N_obs, N_obs))
    for k in range(N_obs):
        for m in range(N_obs):
            i_k = observed_indices[k]
            i_m = observed_indices[m]
            w = localization_weight(i_k, i_m)
            P_yy[k, m] = P_yy_raw[k, m] * w
            
    # Add observation noise variance to the diagonal of P_yy
    P_yy += observation_variance * np.eye(N_obs)
    
    # 5. Compute Kalman Gain
    # K = P_xy P_yy^{-1}
    # Use solve for stability
    try:
        K = np.linalg.solve(P_yy.T, P_xy.T).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        K = P_xy @ np.linalg.pinv(P_yy)
        
    # 6. Update Ensemble
    # Innovation: y_obs - H(mean)
    # H(mean) selects observed indices from the mean
    mean_obs = mean[observed_indices]
    innovation = observations - mean_obs
    
    # Update mean
    mean_new = mean + K @ innovation
    
    # Update perturbations
    # x'_new = x'_old + K (y'_obs)
    # y'_obs is the perturbation of the observations?
    # In EnKF, the observation perturbation is usually sampled or derived from the ensemble if synthetic.
    # Here, we have a single observation vector.
    # The standard update for the full ensemble matrix X (N_members x N_states):
    # X_new = X_old + K (y_obs^T - (H X_old)^T)
    # This is equivalent to updating the mean and perturbations separately if we define the perturbation update correctly.
    # Let's do it directly on the ensemble matrix to avoid confusion.
    
    # H X_old: Select observed columns from ensemble
    ensemble_obs = ensemble[:, observed_indices]
    
    # Innovation for each member: y_obs - ensemble_obs[:, k]
    # y_obs is (N_obs,), ensemble_obs is (N_members, N_obs)
    # We need to subtract the ensemble observed values from the observation vector.
    # innovation_matrix = y_obs[None, :] - ensemble_obs  (N_members, N_obs)
    innovation_matrix = observations[None, :] - ensemble_obs
    
    # Update: X_new = X_old + K @ innovation_matrix.T ?
    # K is (N_states, N_obs)
    # innovation_matrix is (N_members, N_obs)
    # K @ innovation_matrix.T is (N_states, N_members)
    # We need (N_members, N_states)
    # So: X_new = X_old + (K @ innovation_matrix.T).T
    # (K @ innovation_matrix.T).T = innovation_matrix @ K.T
    
    ensemble_new = ensemble + innovation_matrix @ K.T
    
    # Ensure finiteness and bounds
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    return ensemble_new
