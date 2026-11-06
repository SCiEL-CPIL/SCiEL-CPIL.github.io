"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a particle-filter-like approach with
    kernel density estimation for the posterior mean.
    
    Parameters:
    -----------
    ensemble : (16, 20) array of ensemble members
    observations : (10,) array of noisy observations at observed_indices
    observed_indices : (10,) array of indices that are observed
    observation_variance : scalar variance of observation noise
    
    Returns:
    --------
    updated ensemble : (16, 20) array
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    N, D = ensemble.shape
    obs_idx = observed_indices
    obs_var = observation_variance
    obs = observations
    
    # Ensure observations are 1D
    obs = np.asarray(obs).flatten()
    
    # Compute the innovation (observation minus ensemble mean at observed indices)
    ens_mean = np.mean(ensemble, axis=0)
    innovation = obs - ens_mean[obs_idx]
    
    # Compute ensemble covariance at observed indices
    centered = ensemble[:, obs_idx] - ens_mean[obs_idx]
    cov_obs = (centered.T @ centered) / (N - 1) if N > 1 else np.zeros((len(obs_idx), len(obs_idx)))
    
    # Ensure covariance is positive definite
    cov_obs = cov_obs + 1e-10 * np.eye(len(obs_idx))
    
    # Compute the Kalman gain (linear part)
    try:
        cov_obs_inv = np.linalg.inv(cov_obs)
    except np.linalg.LinAlgError:
        cov_obs_inv = np.linalg.pinv(cov_obs)
    
    # Cross-covariance between all states and observed states
    centered_all = ensemble - ens_mean
    cross_cov = (centered_all.T @ centered) / (N - 1) if N > 1 else np.zeros((D, len(obs_idx)))
    
    # Kalman gain
    K = cross_cov @ cov_obs_inv  # (D, len(obs_idx))
    
    # Linear update
    updated_linear = ensemble + K @ innovation[np.newaxis, :]
    
    # Now apply a nonlinear correction using importance weighting
    # Compute observation likelihood for each ensemble member
    # Likelihood: N(observations; ensemble_member[obs_idx], observation_variance)
    
    # Compute squared distances at observed indices
    diffs = ensemble[:, obs_idx] - obs[np.newaxis, :]  # (N, len(obs_idx))
    sq_dists = np.sum(diffs ** 2, axis=1)  # (N,)
    
    # Gaussian likelihood (unnormalized)
    log_lik = -0.5 * sq_dists / obs_var
    log_lik -= 0.5 * len(obs_idx) * np.log(2 * np.pi * obs_var)
    
    # Normalize log-likelihoods
    max_log_lik = np.max(log_lik)
    weights = np.exp(log_lik - max_log_lik)
    weights = weights / np.sum(weights)
    
    # Weighted mean (posterior mean estimate)
    weighted_mean = np.sum(ensemble * weights[:, np.newaxis], axis=0)
    
    # Combine linear Kalman update with weighted mean
    # Use the weighted mean as a better estimate of the posterior
    # Blend: use weighted mean for the state estimate
    
    # Compute the difference between weighted mean and ensemble mean
    correction = weighted_mean - ens_mean
    
    # Apply the correction to all ensemble members, but preserve ensemble spread
    # Scale the correction to maintain reasonable spread
    ens_spread = np.std(ensemble, axis=0)
    
    # Determine how much to nudge based on the relative magnitude
    # Use a soft threshold to avoid over-correcting
    alpha = 0.8  # blending parameter
    
    # Apply the nonlinear correction
    updated_nonlinear = ensemble + alpha * correction[np.newaxis, :]
    
    # Check for non-finite values and clip to [-100, 100]
    updated_nonlinear = np.where(np.isfinite(updated_nonlinear), updated_nonlinear, ens_mean[np.newaxis, :])
    updated_nonlinear = np.clip(updated_nonlinear, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return updated_nonlinear
    
    # Optional: apply a second-order correction using the covariance structure
    # This is a simplified version of the ensemble Kalman filter with inflation
    
    # Inflation to maintain ensemble spread
    inflation_factor = 1.02
    updated_nonlinear = ens_mean[np.newaxis, :] + inflation_factor * (updated_nonlinear - ens_mean[np.newaxis, :])
    
    # Final clipping
    updated_nonlinear = np.clip(updated_nonlinear, -100.0, 100.0)
    updated_nonlinear = np.where(np.isfinite(updated_nonlinear), updated_nonlinear, ens_mean[np.newaxis, :])
    
    return updated_nonlinear
# EVOLVE-BLOCK-END
