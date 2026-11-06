"""Ensemble Kalman Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis step with localization and multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 variables each
    observations : (16, 10) array - noisy observations of 10 even-indexed variables
    observed_indices : array of 10 indices - which variables are observed (even indices)
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - analysis ensemble
    """
    start_time = time.time()
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    ens_centered = ensemble - ens_mean  # (n_members, n_vars)
    
    # Background covariance: C_b = (1/(N-1)) * X^T X
    # We work with the innovation and use the Kalman gain formula
    # For EnKF: K = C_b H^T (H C_b H^T + R)^{-1}
    
    # H is a selection matrix: H @ x gives observed components
    # H C_b H^T is the covariance of observed variables
    obs_ens = ens_centered[:, observed_indices]  # (n_members, n_obs)
    
    # Observation covariance R = observation_variance * I
    R = observation_variance * np.eye(n_obs)
    
    # Compute H C_b H^T = (1/(N-1)) * obs_ens^T @ obs_ens
    S = (obs_ens.T @ obs_ens) / (n_members - 1)  # (n_obs, n_obs)
    S = S + R  # Innovation covariance
    
    # Compute C_b H^T = (1/(N-1)) * ens_centered^T @ obs_ens
    CH = (ens_centered.T @ obs_ens) / (n_members - 1)  # (n_vars, n_obs)
    
    # Kalman gain K = CH @ S^{-1}
    # Use Cholesky or solve for stability
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = CH @ S_inv  # (n_vars, n_obs)
    
    # Innovation: d = observations - H @ x_b
    # x_b is the background ensemble
    obs_background = ensemble[:, observed_indices]  # (n_members, n_obs)
    
    # Analysis update for each member
    # x_a = x_b + K @ (y - H x_b)
    # But we need to apply localization to avoid spurious correlations
    
    # Localization: use a simple exponential decay based on index distance
    # In Lorenz-96 with periodic boundaries, distance is circular
    localization_radius = 5.0  # tune this
    
    # Build localization weights for the gain
    # K is (n_vars, n_obs), we want to damp K[i,j] based on distance between i and observed_indices[j]
    loc_weights = np.ones((n_vars, n_obs))
    for j, obs_idx in enumerate(observed_indices):
        for i in range(n_vars):
            # Circular distance
            dist = abs(i - obs_idx)
            dist = min(dist, n_vars - dist)
            # Exponential localization
            loc_weights[i, j] = np.exp(-dist / localization_radius)
    
    K_loc = K * loc_weights
    
    # Apply analysis update
    updated_ensemble = ensemble.copy()
    for m in range(n_members):
        innovation = observations[m, :] - obs_background[m, :]  # (n_obs,)
        update = K_loc @ innovation  # (n_vars,)
        updated_ensemble[m, :] += update
    
    # Multiplicative inflation to prevent ensemble collapse
    # Compute spread before and after, adjust if needed
    # Simple approach: inflate by a fixed factor based on innovation magnitude
    
    # Adaptive inflation: compute the ratio of background to analysis spread
    bg_spread = np.std(ensemble, axis=0)
    an_spread = np.std(updated_ensemble, axis=0)
    
    # Target: maintain spread comparable to background
    # Inflate where analysis spread is too small
    min_spread = 0.5  # minimum spread to maintain
    for i in range(n_vars):
        if an_spread[i] < min_spread:
            # Inflate this variable
            mean_i = np.mean(updated_ensemble[:, i])
            scale = min_spread / max(an_spread[i], 1e-10)
            updated_ensemble[:, i] = mean_i + (updated_ensemble[:, i] - mean_i) * scale
    
    # Ensure finite and within bounds
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 100:  # Safety margin
        pass  # Already returned a result
    
    return updated_ensemble
# EVOLVE-BLOCK-END
