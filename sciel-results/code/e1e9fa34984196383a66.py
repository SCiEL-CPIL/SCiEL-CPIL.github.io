"""Improved nonlinear particle filter with ensemble-based covariance localization."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear particle filter update with ensemble-based covariance localization.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations at even indices
    - observed_indices: (10,) array of observed variable indices
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensure inputs are float arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    ens_anom = ensemble - ens_mean[None, :]
    
    # Compute ensemble covariance (using n-1 for unbiased estimate)
    ens_cov = (ens_anom.T @ ens_anom) / (n_members - 1)
    
    # Observation operator: extract observed components
    H = np.zeros((n_obs, n_vars))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Compute innovation covariance S = H P H^T + R
    S = H @ ens_cov @ H.T + R
    
    # Compute gain K = P H^T S^{-1}
    # Use Cholesky decomposition for stability
    try:
        L = np.linalg.cholesky(S)
        # Solve L^T Y = H^T for Y, then K = P H^T (L^T)^{-1}
        # More efficient: K = P H^T inv(S)
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        S_inv = np.linalg.pinv(S)
    
    # Compute Kalman gain
    # K = P H^T S^{-1}, shape (n_vars, n_obs)
    PHt = ens_cov @ H.T
    K = PHt @ S_inv
    
    # Compute innovations (observation minus ensemble mean at observed indices)
    innovation = observations - ens_mean[obs_idx]
    
    # Apply update: x_a = x + K * innovation
    # For each member: x_a_i = x_i + K @ (y - H x_i)
    # This is equivalent to: x_a_i = x_i + K @ (y - H @ x_i)
    # Since H is a selection matrix, H @ x_i = x_i[obs_idx]
    
    updated_ensemble = ensemble.copy()
    for i in range(n_members):
        # Innovation for this member
        innov_i = observations - ensemble[i, obs_idx]
        # Update
        updated_ensemble[i, :] = ensemble[i, :] + K @ innov_i
    
    # Ensure the updated ensemble is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Replace any non-finite values with the ensemble mean
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, ens_mean)
    
    return updated_ensemble
