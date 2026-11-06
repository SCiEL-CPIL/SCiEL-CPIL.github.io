"""Weighted ensemble Kalman filter with localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Weighted ensemble analysis update for Lorenz-96.
    
    Uses importance weighting based on observation likelihood,
    with spatial localization and ensemble spread maintenance.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    DEADLINE = t0 + 145.0
    
    n_particles, n_dim = ensemble.shape
    obs_var = observation_variance
    n_obs = len(observed_indices)
    
    # Compute log-likelihood weights for each particle
    log_weights = np.zeros(n_particles)
    
    for i in range(n_particles):
        state = ensemble[i]
        log_lik = 0.0
        for j, obs_idx in enumerate(observed_indices):
            residual = observations[j] - state[obs_idx]
            log_lik += -0.5 * (residual ** 2) / obs_var
        log_weights[i] = log_lik
    
    # Normalize to get importance weights
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    weights /= np.sum(weights)
    
    # Effective sample size check
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is too low, use uniform weights (avoid particle degeneracy)
    if ess < n_particles * 0.3:
        weights = np.ones(n_particles) / n_particles
    
    # Compute weighted mean
    mean = np.sum(ensemble * weights[:, np.newaxis], axis=0)
    
    # Compute weighted covariance with localization
    # Use a banded covariance structure
    L = 2  # localization range
    
    # Weighted deviations
    dev = (ensemble - mean[np.newaxis, :]) * np.sqrt(weights[:, np.newaxis])
    
    # Weighted covariance
    cov = np.dot(dev.T, dev) / (n_particles - 1)
    
    # Apply localization: zero out correlations beyond range L
    for i in range(n_dim):
        for j in range(n_dim):
            # Circular distance for periodic system
            dist = min(abs(i - j), n_dim - abs(i - j))
            if dist > L:
                cov[i, j] = 0.0
    
    # Ensure positive semi-definite
    cov = (cov + cov.T) / 2
    cov += 1e-6 * np.eye(n_dim)
    
    # Compute Kalman gain
    # Observation operator: observes even indices
    H = np.zeros((n_obs, n_dim))
    for j, obs_idx in enumerate(observed_indices):
        H[j, obs_idx] = 1.0
    
    # Innovation covariance
    R = obs_var * np.eye(n_obs)
    S = np.dot(H, np.dot(cov, H.T)) + R
    
    # Kalman gain
    K = np.dot(cov, H.T)
    try:
        K = np.dot(K, np.linalg.inv(S))
    except np.linalg.LinAlgError:
        K = np.dot(K, np.linalg.pinv(S))
    
    # Innovation
    innovation = observations - np.dot(H, mean)
    
    # Analysis mean
    mean_analysis = mean + np.dot(K, innovation)
    
    # Analysis covariance
    A = np.eye(n_dim) - K @ H
    cov_analysis = A @ cov @ A.T + K @ R @ K.T
    
    # Ensure positive semi-definite
    cov_analysis = (cov_analysis + cov_analysis.T) / 2
    cov_analysis += 1e-6 * np.eye(n_dim)
    
    # Cholesky decomposition for sampling
    try:
        L_chol = np.linalg.cholesky(cov_analysis)
    except np.linalg.LinAlgError:
        # Use eigenvalue decomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh(cov_analysis)
        eigvals = np.maximum(eigvals, 1e-10)
        L_chol = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate new ensemble members
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility within call
    result = np.zeros((n_particles, n_dim))
    for i in range(n_particles):
        z = rng.standard_normal(n_dim)
        result[i] = mean_analysis + z @ L_chol.T
    
    # Ensure finite and within bounds
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > DEADLINE:
        return result
    
    # Maintain minimum spread to prevent collapse
    for d in range(n_dim):
        spread = np.std(result[:, d])
        if spread < 0.5:
            noise = rng.normal(0, 0.5, n_particles)
            result[:, d] += noise
    
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
