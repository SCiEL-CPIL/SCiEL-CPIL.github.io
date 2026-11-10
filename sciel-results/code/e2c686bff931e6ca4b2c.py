"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a localized Ensemble Kalman Filter (LEKF)
    with adaptive multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed coordinates (even indices).
    observed_indices : ndarray, shape (10,)
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Convert to arrays for safety
    ensemble = np.array(ensemble, dtype=float)
    observations = np.array(observations, dtype=float)
    observed_indices = np.array(observed_indices, dtype=int)
    
    # Step 1: Compute ensemble mean and anomaly matrix
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Step 2: Localize the forecast error covariance using a Gaspari-Cohn function
    # Distance between variables in the periodic Lorenz-96 system
    def gc_localization(dist, cutoff=8.0):
        """Gaspari-Cohn correlation function."""
        dist = np.abs(dist)
        if dist <= cutoff / 2.0:
            return (-5.0 / 4.0 * dist**5 / cutoff**5 + 5.0 / 2.0 * dist**4 / cutoff**4 -
                    30.0 / 5.0 * dist**3 / cutoff**3 + 5.0 * dist**2 / cutoff**2 +
                    4.0 - 2.0 / 3.0)
        elif dist <= cutoff:
            return (5.0 / 4.0 * dist**5 / cutoff**5 - 5.0 / 2.0 * dist**4 / cutoff**4 +
                    30.0 / 5.0 * dist**3 / cutoff**3 - 5.0 * dist**2 / cutoff**2 +
                    5.0 - 2.0 / 3.0)
        else:
            return 0.0
    
    # Build localization matrix for observation errors
    # For each observed variable, compute distances to all state variables
    loc_matrix = np.ones((n_obs, n_vars))
    for i, obs_idx in enumerate(observed_indices):
        for j in range(n_vars):
            # Periodic distance
            dist = abs(obs_idx - j)
            dist = min(dist, n_vars - dist)
            loc_matrix[i, j] = gc_localization(dist, cutoff=8.0)
    
    # Step 3: Compute the Kalman gain using the ensemble anomalies
    # Observation anomaly matrix: (n_obs, n_members)
    obs_anomalies = anomalies[:, observed_indices].T  # shape (n_obs, n_members)
    
    # Apply localization to the cross-covariance
    # Cross-covariance: (n_vars, n_obs) = anomalies @ obs_anomalies.T / (n_members - 1)
    cross_cov = anomalies @ obs_anomalies.T  # (n_vars, n_obs)
    
    # Localize the cross-covariance
    cross_cov_loc = cross_cov * loc_matrix.T  # (n_vars, n_obs)
    
    # Observation error covariance: (n_obs, n_obs)
    # Include localization in the observation error covariance
    obs_cov = np.eye(n_obs) * observation_variance
    
    # Add localization effect to observation covariance
    # The effective observation error variance is increased by localization
    # We use a simple approach: add the diagonal of the localization effect
    for i in range(n_obs):
        obs_cov[i, i] += (1.0 - loc_matrix[i, observed_indices[i]]) * observation_variance * 0.5
    
    # Compute the Kalman gain: K = cross_cov_loc @ inv(obs_cov + obs_anomalies @ obs_anomalies.T / (n_members - 1))
    # obs_anomalies @ obs_anomalies.T / (n_members - 1) is the sample observation error covariance
    obs_sample_cov = obs_anomalies @ obs_anomalies.T / (n_members - 1)  # (n_obs, n_obs)
    
    # Total observation error covariance
    S = obs_sample_cov + obs_cov  # (n_obs, n_obs)
    
    # Compute K = cross_cov_loc @ S^{-1}
    # Use solve for stability
    try:
        K = np.linalg.solve(S.T, cross_cov_loc.T).T  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = cross_cov_loc @ np.linalg.pinv(S)
    
    # Step 4: Compute the analysis increment
    # Innovation: (n_obs,)
    innovation = observations - ens_mean[observed_indices]
    
    # Analysis increment: (n_vars,)
    increment = K @ innovation
    
    # Step 5: Apply adaptive multiplicative inflation
    # Estimate the inflation factor based on the ratio of observed error to expected error
    # Expected observation error variance
    expected_obs_err_var = np.mean(np.diag(obs_sample_cov + obs_cov))
    actual_obs_err_var = np.mean(innovation ** 2)
    
    # Inflation factor (square root of the ratio, capped)
    if expected_obs_err_var > 0:
        inflation_factor = np.sqrt(actual_obs_err_var / expected_obs_err_var)
    else:
        inflation_factor = 1.0
    
    # Cap inflation to avoid instability
    inflation_factor = np.clip(inflation_factor, 1.0, 2.0)
    
    # Apply inflation to the anomalies
    inflated_anomalies = inflation_factor * anomalies
    
    # Step 6: Update the ensemble
    # New mean = old mean + increment
    new_mean = ens_mean + increment
    
    # New anomalies = inflated anomalies
    new_anomalies = inflated_anomalies
    
    # Reconstruct the ensemble
    updated_ensemble = new_mean + new_anomalies
    
    # Ensure finite values and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
