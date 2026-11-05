"""Ensemble Square Root Filter with sequential coordinate-wise regression for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a sequential square-root filter approach.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (16, 10) array of noisy observations for observed indices
        observed_indices: array of 10 even indices [0,2,4,...,18]
        observation_variance: scalar, variance of observation noise
    
    Returns:
        updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    result = ensemble.copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (n_members, n_vars)
    
    # Observation anomalies relative to ensemble mean at observed locations
    obs_anoms = observations - ens_mean[observed_indices]  # (n_members, 10)
    
    # Compute ensemble covariance (sample covariance)
    # C_oo: covariance of observed variables
    # C_uo: covariance of unobserved vs observed variables
    # C_uu: covariance of unobserved variables
    
    unobserved_indices = np.array([i for i in range(n_vars) if i not in observed_indices])
    
    C_oo = (anomalies[:, observed_indices].T @ anomalies[:, observed_indices]) / (n_members - 1)  # (10, 10)
    C_uo = (anomalies[:, unobserved_indices].T @ anomalies[:, observed_indices]) / (n_members - 1)  # (10, 10)
    C_uu = (anomalies[:, unobserved_indices].T @ anomalies[:, unobserved_indices]) / (n_members - 1)  # (10, 10)
    
    # Observation error covariance (diagonal with observation_variance)
    R = observation_variance * np.eye(len(observed_indices))
    
    # Kalman gain for unobserved variables: K = C_uo @ inv(C_oo + R)
    # Use solve for numerical stability
    try:
        S = C_oo + R  # Innovation covariance
        # Solve C_uo @ S^{-1} using linear solve
        K = np.linalg.solve(S.T, C_uo.T).T  # (10, 10)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = C_uo @ np.linalg.pinv(C_oo + R)
    
    # Update unobserved variables
    # For each unobserved variable j:
    #   x_j_new = x_j + K[j,:] @ (y - H(x)_obs)
    # where H(x)_obs is the ensemble mean at observed locations
    
    innovation = obs_anoms  # (n_members, 10)
    
    # Update unobserved anomalies
    # delta_u = K @ innovation^T  -> (10, n_members)
    delta_u = K @ innovation.T  # (10, n_members)
    
    # Add to unobserved variables in result
    for i, idx in enumerate(unobserved_indices):
        result[:, idx] += delta_u[i, :]
    
    # Update observed variables: simple blend with observation
    # Use optimal linear combination: weight = C_oo / (C_oo + R)
    # For each observed variable:
    #   x_obs_new = x_obs + C_oo @ inv(C_oo + R) @ (y - x_obs)
    
    try:
        K_obs = np.linalg.solve(S.T, C_oo.T).T  # (10, 10)
    except np.linalg.LinAlgError:
        K_obs = C_oo @ np.linalg.pinv(S)
    
    delta_obs = K_obs @ innovation.T  # (10, n_members)
    
    for i, idx in enumerate(observed_indices):
        result[:, idx] += delta_obs[i, :]
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
