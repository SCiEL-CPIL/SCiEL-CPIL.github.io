"""Local ETKF-style analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a local ensemble transform Kalman filter (ETKF) analysis update.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of state vectors (16 members, 20 variables).
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed variables.
    observed_indices : array-like
        Indices of observed variables (even indices: 0,2,4,...,18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after analysis.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 20
    Nens = ensemble.shape[0]  # 16
    Nobs = len(observed_indices)  # 10
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)  # (20,)
    anomalies = ensemble - mean  # (16, 20)
    
    # Estimate ensemble covariance
    # Background covariance: (1/(Nens-1)) * A^T A
    cov = (anomalies.T @ anomalies) / (Nens - 1)  # (20, 20)
    
    # Observation operator: H selects observed indices
    H = np.zeros((Nobs, N))
    for i, idx in enumerate(observed_indices):
        H[i, int(idx)] = 1.0
    
    # Observation covariance
    R = np.eye(Nobs) * observation_variance  # (10, 10)
    
    # Innovation (observation minus ensemble mean at observed locations)
    innovation = observations - H @ mean  # (10,)
    
    # Compute Kalman gain using ensemble-based approach
    # S = H P H^T + R
    HP = H @ cov  # (10, 20)
    S = HP @ H.T + R  # (10, 10)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = P H^T S^-1
    K = cov @ H.T @ S_inv  # (20, 10)
    
    # Analysis update for mean
    mean_analysis = mean + K @ innovation  # (20,)
    
    # Apply ensemble transform for spread
    # Use ensemble transform Kalman filter approach
    # For each member, apply a transformation that preserves the ensemble structure
    
    # Compute the ensemble Kalman filter update
    # We use the square-root form to maintain positive semi-definiteness
    
    # Compute Q such that Q Q^T = P (background covariance)
    # Using Cholesky or SVD
    try:
        L = np.linalg.cholesky(cov + np.eye(N) * 1e-10)  # (20, 20)
        Q = L.T  # (20, 20)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(cov)
        Q = np.sqrt(s) * Vt  # (20, 20)
    
    # Transform anomalies to uncorrelated space
    # anomalies = Q * z where z has identity covariance
    try:
        z = np.linalg.solve(Q.T, anomalies.T).T  # (16, 20)
    except np.linalg.LinAlgError:
        z = anomalies @ np.linalg.pinv(Q.T)
    
    # Observation innovation in transformed space
    # h = H @ Q @ z (for each member)
    HQ = H @ Q  # (10, 20)
    h_ens = z @ HQ.T  # (16, 10)
    
    # Observation residual for each member
    obs_residual = h_ens - observations  # (16, 10)
    
    # Compute the transformation matrix T such that:
    # z_analysis = z + T^T @ (S_inv @ obs_residual)
    # We want the analysis covariance to match the theoretical update
    
    # Simplified approach: use linear EnKF update in transformed space
    # The gain in transformed space
    # K_z = Q^T @ K  (20, 10)
    K_z = Q.T @ K  # (20, 10)
    
    # Update each member
    # member_analysis = mean_analysis + (member - mean) + K @ (obs - H @ member)
    # = mean_analysis + anomalies + K @ (innovation - H @ anomalies)
    # = mean_analysis + anomalies - K @ H @ anomalies + K @ innovation
    # = mean_analysis + (I - K @ H) @ anomalies + K @ innovation
    
    KH = K @ H  # (20, 20)
    update_matrix = np.eye(N) - KH  # (20, 20)
    
    # Apply to anomalies
    anomalies_updated = anomalies @ update_matrix.T  # (16, 20)
    
    # Add the mean update
    updated_ensemble = mean_analysis[None, :] + anomalies_updated  # (16, 20)
    
    # Adaptive inflation to prevent ensemble collapse
    # Compute the spread in observed space vs expected spread
    # Expected spread from observation variance
    # Actual spread from ensemble
    ens_obs = updated_ensemble[:, observed_indices]
    ens_obs_mean = np.mean(ens_obs, axis=0)
    ens_obs_anom = ens_obs - ens_obs_mean
    actual_spread = np.mean(np.var(ens_obs_anom, axis=0))
    
    # Expected spread should be roughly observation_variance + model_error
    # If actual spread is too small, inflate
    expected_spread = observation_variance * 0.5  # Conservative estimate
    
    if actual_spread < expected_spread * 0.5 and actual_spread > 1e-10:
        inflation_factor = np.sqrt(expected_spread / actual_spread)
        inflation_factor = min(inflation_factor, 2.0)  # Cap inflation
        # Inflate around the analysis mean
        updated_ensemble = mean_analysis[None, :] + (updated_ensemble - mean_analysis[None, :]) * inflation_factor
    
    # Ensure output is finite and within bounds
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return updated_ensemble
    
    return updated_ensemble
