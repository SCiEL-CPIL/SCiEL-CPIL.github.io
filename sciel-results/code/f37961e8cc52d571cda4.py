"""Lorenz-96 particle filter assimilation with KDE posterior and adaptive inflation."""
import numpy as np
from scipy.stats import norm
import time

# Module-level state for adaptive inflation
_inflation_factor = 1.0
_inflation_history = []

def _lorenz96_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    def f(x):
        n = len(x)
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _lorenz96_propagate(x, dt, F, steps):
    """Propagate Lorenz-96 state for given number of steps."""
    for _ in range(steps):
        x = _lorenz96_step(x, dt, F)
    return x

def _ring_distance(i, j, n):
    """Circulant ring distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)

def _localization_weight(dist, L):
    """Gaussian localization weight based on ring distance."""
    return np.exp(-0.5 * (dist / L) ** 2)

def _kde_posterior_mean(samples, observations, obs_idx, obs_var, n_vars, L):
    """
    Compute posterior mean using kernel density estimation approach.
    For each unobserved variable, estimate its distribution given observations
    using weighted kernel density.
    """
    n_ens = len(samples)
    posterior_mean = np.zeros(n_vars)
    
    # Compute innovation-based weights for each ensemble member
    # For each member, compute the likelihood of observations
    weights = np.zeros(n_ens)
    
    for k in range(n_ens):
        log_lik = 0.0
        for j, obs_i in enumerate(obs_idx):
            pred = samples[k, obs_i]
            innov = observations[j] - pred
            # Gaussian likelihood
            log_lik += -0.5 * (innov / np.sqrt(obs_var)) ** 2
        weights[k] = log_lik
    
    # Normalize weights (softmax-like for stability)
    max_w = np.max(weights)
    weights = np.exp(weights - max_w)
    weights /= np.sum(weights)
    
    # For observed variables: weighted average
    for obs_i in obs_idx:
        posterior_mean[obs_i] = np.sum(weights * samples[:, obs_i])
    
    # For unobserved variables: use localization-weighted KDE
    unobs_idx = [i for i in range(n_vars) if i not in obs_idx]
    for u in unobs_idx:
        # Collect contributions from all ensemble members, weighted by
        # their observation likelihood and localization to nearby observed points
        vals = np.zeros(n_ens)
        loc_weights = np.zeros(n_ens)
        
        for k in range(n_ens):
            # Base weight from observation likelihood
            base_w = weights[k]
            
            # Localization: weight by distance to nearest observed indices
            min_dist = np.inf
            for obs_i in obs_idx:
                d = _ring_distance(u, obs_i, n_vars)
                min_dist = min(min_dist, d)
            
            loc_w = _localization_weight(min_dist, L)
            vals[k] = samples[k, u]
            loc_weights[k] = base_w * loc_w
        
        total_w = np.sum(loc_weights)
        if total_w > 1e-15:
            posterior_mean[u] = np.sum(loc_weights * vals) / total_w
        else:
            # Fallback: unweighted ensemble mean
            posterior_mean[u] = np.mean(samples[:, u])
    
    return posterior_mean, weights

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter assimilation update for Lorenz-96.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed states
        observed_indices: indices of observed variables (even indices)
        observation_variance: variance of observation noise
    
    Returns:
        Updated ensemble with same (16, 20) shape
    """
    global _inflation_factor, _inflation_history
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_ens, n_vars = ensemble.shape
    obs_var = observation_variance
    obs_idx = list(observed_indices)
    
    # Inflation: adaptively adjust ensemble spread
    # Compute innovation variance to assess if inflation is needed
    innov_var = np.zeros(len(obs_idx))
    for j, obs_i in enumerate(obs_idx):
        innov = observations[j] - ensemble[:, obs_i]
        innov_var[j] = np.var(innov)
    
    mean_innov_var = np.mean(innov_var)
    expected_innov_var = obs_var + np.mean(np.var(ensemble[:, obs_idx], axis=0))
    
    # Adaptive inflation: increase if innovations too large, decrease if too small
    ratio = mean_innov_var / max(expected_innov_var, 1e-10)
    if ratio > 1.2:
        _inflation_factor = min(_inflation_factor * 1.05, 2.0)
    elif ratio < 0.8:
        _inflation_factor = max(_inflation_factor * 0.95, 0.8)
    
    # Apply multiplicative inflation to ensemble
    inflated_ensemble = ensemble.copy()
    for k in range(n_ens):
        mean_state = np.mean(ensemble, axis=0)
        inflated_ensemble[k] = mean_state + _inflation_factor * (ensemble[k] - mean_state)
    
    # Localization length scale
    L = 2.5
    
    # Compute posterior mean using KDE-based particle filter
    posterior_mean, particle_weights = _kde_posterior_mean(
        inflated_ensemble, observations, obs_idx, obs_var, n_vars, L
    )
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # Return simple nudged result if running out of time
        result = inflated_ensemble.copy()
        for j, obs_i in enumerate(obs_idx):
            result[:, obs_i] += 0.5 * (observations[j] - result[:, obs_i])
        return result
    
    # Resample particles using systematic resampling
    cumulative_weights = np.cumsum(particle_weights)
    cumulative_weights /= cumulative_weights[-1]
    
    u0 = np.random.uniform(0, 1.0 / n_ens)
    u = u0 + np.arange(n_ens) / n_ens
    
    resampled_indices = np.searchsorted(cumulative_weights, u)
    resampled_indices = np.clip(resampled_indices, 0, n_ens - 1)
    
    resampled_ensemble = inflated_ensemble[resampled_indices]
    
    # Blend resampled particles with posterior mean to ensure good spread
    # and accuracy
    blend_alpha = 0.3  # Fraction of posterior mean to blend in
    
    result = np.zeros_like(resampled_ensemble)
    for k in range(n_ens):
        result[k] = (1 - blend_alpha) * resampled_ensemble[k] + blend_alpha * posterior_mean
    
    # Ensure observed variables are well-represented
    for j, obs_i in enumerate(obs_idx):
        result[:, obs_i] = 0.7 * result[:, obs_i] + 0.3 * observations[j]
    
    # Clip to valid range
    result = np.clip(result, -100, 100)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
