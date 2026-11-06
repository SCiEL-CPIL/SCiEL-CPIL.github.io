"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.90  # Use 90% of the 150s budget

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _compute_localization_weights(D, observed_indices, localization_length):
    """
    Pre-compute localization weights for cross-covariance.
    Returns a (D, M) matrix of weights.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    return weights


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a Nonlinear Particle Filter with KDE Resampling.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        # Deadline exceeded, return current ensemble as-is
        return np.clip(ensemble, -100.0, 100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # 1. Compute Importance Weights
    Hx = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - Hx  # (N, M)
    
    # Log-likelihood: -0.5 * sum((y - Hx)^2 / sigma^2)
    innov_sq = np.sum(innovations ** 2, axis=1)
    log_weights = -0.5 * innov_sq / observation_variance
    
    # Normalize weights (log-sum-exp for stability)
    max_log_w = np.max(log_weights)
    weights = np.exp(log_weights - max_log_w)
    sum_w = np.sum(weights)
    if sum_w < 1e-10:
        weights = np.ones(N) / N
    else:
        weights /= sum_w
    
    # 2. Kernel Density Estimate (KDE) Sampling
    # Estimate bandwidth based on ensemble spread
    ens_std = np.std(ensemble, axis=0)
    
    # Adaptive noise scale: larger if innovations are large (model-data mismatch)
    mean_innov_sq = np.mean(innov_sq)
    mismatch_factor = 1.0 + 0.5 * (mean_innov_sq / observation_variance)
    noise_scale = 0.5 * ens_std * mismatch_factor
    
    # Systematic Resampling to pick parents
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0  # Ensure exact sum
    
    u = np.random.uniform(0, 1.0 / N)
    indices = np.zeros(N, dtype=int)
    for i in range(N):
        u_i = u + i / N
        indices[i] = np.searchsorted(cum_weights, u_i)
    
    # Resampled ensemble
    resampled = ensemble[indices]
    
    # Add Gaussian noise to simulate KDE sampling
    noise = np.random.normal(0, 1, size=(N, D)) * noise_scale
    new_ensemble = resampled + noise
    
    # 3. Mean Correction
    # The KDE sample mean may be biased. We want the new ensemble mean
    # to be the weighted average of the prior ensemble (posterior mean approx).
    posterior_mean = np.average(ensemble, axis=0, weights=weights)
    
    # Current new mean
    new_mean = np.mean(new_ensemble, axis=0)
    
    # Shift the new ensemble to match the posterior mean
    new_ensemble = new_ensemble + (posterior_mean - new_mean)
    
    # 4. Safety Checks
    new_ensemble = np.nan_to_num(new_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    return new_ensemble

# EVOLVE-BLOCK-END
