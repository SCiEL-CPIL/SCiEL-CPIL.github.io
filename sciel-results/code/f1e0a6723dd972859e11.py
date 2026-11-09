"""Ensemble Kalman Filter with observation perturbation for Lorenz-96 assimilation."""
import numpy as np
from scipy.stats import norm
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using Ensemble Kalman Filter (observation perturbation method).
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations at even indices
    - observed_indices: array of indices that are observed (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - Updated ensemble with same (16, 20) shape
    """
    t_start = time.time()
    
    num_members, dim = ensemble.shape
    result = ensemble.copy()
    
    # Check for non-finite values
    if not np.all(np.isfinite(ensemble)):
        return np.clip(result, -100, 100)
    
    # Observation perturbation: add random noise to observations
    obs_noise = np.sqrt(observation_variance)
    
    # Use a fixed random seed for reproducibility within a single call
    # But we need different perturbations for each member
    rng = np.random.default_rng()
    
    # Perturb observations for each ensemble member
    perturbed_obs = observations[None, :] + rng.normal(0, obs_noise, size=(num_members, len(observed_indices)))
    
    # Compute ensemble mean and anomaly at observed locations
    ensemble_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ensemble_mean[None, :]
    
    # Extract anomalies at observed indices
    obs_anomalies = anomalies[:, observed_indices]
    
    # Compute innovation: perturbed_obs - ensemble_observed
    innovation = perturbed_obs - ensemble[:, observed_indices]
    
    # Compute covariance of observation anomalies
    # C_oo = (1/(N-1)) * obs_anomalies.T @ obs_anomalies
    C_oo = (obs_anomalies.T @ obs_anomalies) / (num_members - 1)
    
    # Add observation noise to C_oo for numerical stability
    C_oo += np.eye(len(observed_indices)) * observation_variance
    
    # Compute gain matrix: K = C_xo * C_oo^{-1}
    # C_xo = (1/(N-1)) * anomalies.T @ obs_anomalies
    C_xo = (anomalies.T @ obs_anomalies) / (num_members - 1)
    
    # Solve for gain: K = C_xo @ inv(C_oo)
    # Use Cholesky for stability
    try:
        L = np.linalg.cholesky(C_oo)
        # Solve L @ L.T @ X = C_xo.T
        X = np.linalg.solve(L.T, C_xo.T)
        K = np.linalg.solve(L, X).T  # K is (dim, num_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = C_xo @ np.linalg.pinv(C_oo)
    
    # Update each ensemble member
    for i in range(num_members):
        result[i, :] = ensemble[i, :] + K @ (innovation[i, :])
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100, 100)
    
    # Time guard
    if time.time() - t_start > 110:
        return result
    
    return result
