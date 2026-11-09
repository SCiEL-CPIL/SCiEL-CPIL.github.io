"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96 partial observations.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: indices of observed variables (even indices: 0,2,4,...,18)
    - observation_variance: scalar, variance of observation noise
    
    Returns:
    - Updated ensemble with same (16, 20) shape
    """
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensemble mean
    x_mean = np.mean(ensemble, axis=0)
    
    # Anomalies
    X = ensemble - x_mean
    
    # State covariance: (n_vars, n_vars)
    # C = X.T @ X / (n_members - 1)
    C = (X.T @ X) / (n_members - 1)
    
    # Observation operator: H selects observed indices
    # H @ x_mean = x_mean[observed_indices]
    x_mean_obs = x_mean[observed_indices]
    
    # Innovation (observation minus predicted observation)
    innovation = observations - x_mean_obs  # (n_obs,)
    
    # Observation error covariance: R = obs_var * I
    R = observation_variance * np.eye(n_obs)
    
    # Forecast observation covariance: H C H^T
    C_obs = C[np.ix_(observed_indices, observed_indices)]  # (n_obs, n_obs)
    
    # Innovation covariance: S = H C H^T + R
    S = C_obs + R  # (n_obs, n_obs)
    
    # Kalman gain: K = C H^T S^{-1}  -> shape (n_vars, n_obs)
    # K = C[:, observed_indices] @ inv(S)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    C_Ht = C[:, observed_indices]  # (n_vars, n_obs)
    K = C_Ht @ S_inv  # (n_vars, n_obs)
    
    # Update ensemble mean
    x_mean_new = x_mean + K @ innovation  # (n_vars,)
    
    # Update each ensemble member
    # x_i_new = x_i + K @ (y - H x_i)
    # For each member, innovation_i = observations - x_i[observed_indices]
    innovations_members = observations[None, :] - ensemble[:, observed_indices]  # (n_members, n_obs)
    
    # Update: x_i_new = x_i + K @ innovations_i
    # K @ innovations_i for each member: (n_vars, n_obs) @ (n_members, n_obs).T -> (n_vars, n_members)
    update = (K @ innovations_members.T).T  # (n_members, n_vars)
    
    ensemble_new = ensemble + update
    
    # Inflation: adjust based on innovation statistics
    # Compute observed innovation variance vs expected
    # Expected innovation variance = diag(S)
    S_diag = np.diag(S)
    innov_var = np.mean(innovation ** 2)
    expected_innov_var = np.mean(S_diag)
    
    # Inflation factor: if observed innovations are larger than expected, inflate
    if expected_innov_var > 0:
        ratio = innov_var / expected_innov_var
        # Gentle inflation: sqrt of ratio, capped
        inflation_factor = np.sqrt(min(ratio, 2.0))
        # Apply multiplicative inflation around the new mean
        ensemble_new = x_mean_new + inflation_factor * (ensemble_new - x_mean_new)
    
    # Clip to valid range
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    # Ensure finiteness
    ensemble_new = np.nan_to_num(ensemble_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline check
    if time.time() > deadline:
        # Return current result immediately
        return ensemble_new
    
    return ensemble_new
# EVOLVE-BLOCK-END
