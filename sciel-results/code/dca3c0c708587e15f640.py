"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Kalman Filter (EnKF) analysis step.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each a 20-dimensional state vector.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 observed coordinates (even indices).
    observed_indices : array-like, shape (10,)
        Indices of the observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # shape (n_members, n_state)
    
    # Background covariance (scaled for numerical stability)
    # C_bg = (1/(n-1)) * A^T A
    # We use the standard EnKF formulation
    
    # Observation operator: H maps state to observed coordinates
    # H is sparse: H[i, observed_indices[i]] = 1
    
    # Compute H * anomalies for each member
    H_anom = anomalies[:, observed_indices]  # shape (n_members, n_obs)
    
    # Innovation: observations - H * mean
    innovation = observations - mean[observed_indices]  # shape (n_obs,)
    
    # Compute the Kalman gain using the ensemble formulation
    # S = H * C_bg * H^T + R
    # where C_bg = (1/(n-1)) * A^T A, R = obs_var * I
    
    # S = (1/(n-1)) * H_anom^T H_anom + obs_var * I
    # shape: (n_obs, n_obs)
    
    H_anom_T = H_anom.T  # shape (n_obs, n_members)
    S = (H_anom_T @ H_anom) / (n_members - 1) + observation_variance * np.eye(n_obs)
    
    # Numerical stabilization: add small regularization if needed
    # Ensure S is positive definite
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = C_bg * H^T * S_inv
    # C_bg * H^T = (1/(n-1)) * A^T * H^T = (1/(n-1)) * H_anom^T
    # So K = (1/(n-1)) * H_anom^T * S_inv
    # shape: (n_state, n_obs)
    
    K = (H_anom_T @ S_inv) / (n_members - 1)  # shape (n_state, n_obs)
    
    # Update the mean
    mean_update = K @ innovation  # shape (n_state,)
    new_mean = mean + mean_update
    
    # Update each ensemble member
    # x_i_new = x_i + K * (y - H*x_i)
    # = x_i + K * (innovation - H_anom_i)
    # where H_anom_i = H*x_i - H*mean = H*(x_i - mean)
    
    updated_ensemble = ensemble.copy()
    for i in range(n_members):
        H_anom_i = H_anom[i]  # shape (n_obs,)
        innov_i = innovation - H_anom_i
        updated_ensemble[i] += K @ innov_i
    
    # Adaptive inflation: monitor spread and adjust if necessary
    # Compute spread ratio: observed spread vs expected spread
    # This is a simple form of adaptive inflation
    
    # Compute the analysis spread at observed locations
    analysis_anom = updated_ensemble - np.mean(updated_ensemble, axis=0)
    analysis_spread_obs = np.var(analysis_anom[:, observed_indices], axis=0)  # shape (n_obs,)
    
    # Expected analysis spread: (I - K*H) * C_bg * (I - K*H)^T + ... 
    # Simplified: compare to observation variance
    # If spread is too low, inflate
    
    # Compute a simple inflation factor
    # Target: analysis spread should be at least a fraction of observation variance
    # Use a conservative approach: only inflate if spread is very low
    
    mean_analysis_spread = np.mean(analysis_spread_obs)
    # Expected reduction in variance: roughly obs_var * (1 - 1/n_obs) for perfect obs
    # But with partial obs, it's more complex. Use a heuristic.
    
    # Simple adaptive inflation: if spread is less than 50% of obs_var, inflate
    inflation_factor = 1.0
    if mean_analysis_spread < 0.5 * observation_variance:
        # Inflate anomalies about the mean
        mean_analysis = np.mean(updated_ensemble, axis=0)
        # Compute how much to inflate
        target_spread = 0.7 * observation_variance
        if mean_analysis_spread > 1e-10:
            inflation_factor = np.sqrt(target_spread / mean_analysis_spread)
            inflation_factor = min(inflation_factor, 1.5)  # Cap inflation
            updated_ensemble = mean_analysis + inflation_factor * (updated_ensemble - mean_analysis)
    
    # Ensure finite values and within bounds
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
