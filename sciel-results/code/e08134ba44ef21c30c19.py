"""Ensemble particle filter with KDE posterior for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Lorenz-96 RHS with periodic boundary conditions."""
    n = len(x)
    return (x[np.roll(x, 1, axis=0)] - x[np.roll(x, 2, axis=0)]) * x[np.roll(x, -1, axis=0)] - x + F


def _rk4_step(x, dt, F):
    """Single RK4 step for one state vector."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_step_ensemble(ensemble, dt, F, n_steps=5):
    """Propagate ensemble members by RK4 for n_steps."""
    for _ in range(n_steps):
        for i in range(ensemble.shape[0]):
            ensemble[i] = _rk4_step(ensemble[i], dt, F)
    return ensemble


def _bandwidth_selection(ensemble, observed_indices):
    """Select bandwidth for KDE using Silverman's rule on observed components."""
    n_members = ensemble.shape[0]
    n_obs = len(observed_indices)
    bandwidths = np.zeros(n_obs)
    for j, idx in enumerate(observed_indices):
        vals = ensemble[:, idx]
        std = np.std(vals)
        if std < 1e-8:
            std = 1.0
        # Silverman's rule
        bandwidths[j] = 1.06 * std * n_members ** (-1.0 / 5.0)
    return bandwidths


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update with KDE posterior for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s limit

    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)

    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    obs_var = observation_variance
    obs_std = np.sqrt(obs_var)

    # Estimate F from ensemble mean (use a rough estimate)
    # Lorenz-96: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # At quasi-steady state, mean of (x_{i+1} - x_{i-2}) * x_{i-1} - x_i ≈ -F
    # So F ≈ -mean over i and members of [(x_{i+1} - x_{i-2}) * x_{i-1} - x_i]
    x_mean = np.mean(ensemble, axis=0)
    n = n_state
    f_est = 0.0
    for i in range(n):
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        f_est += (x_mean[ip1] - x_mean[im2]) * x_mean[im1] - x_mean[i]
    F = -f_est / n
    # Clamp F to reasonable range
    F = np.clip(F, 2.0, 15.0)

    # Propagate ensemble forward by 5 RK4 steps at dt=0.01
    dt = 0.01
    n_steps = 5
    for _ in range(n_steps):
        for i in range(n_members):
            ensemble[i] = _rk4_step(ensemble[i], dt, F)

    # Compute likelihoods for each particle based on observations
    # Likelihood: product over observed indices of N(observations[j], obs_var)
    # log-likelihood = -0.5 * sum_j [(obs_j - x_i,obs_j)^2 / obs_var]
    log_likelihoods = np.zeros(n_members)
    for j, idx in enumerate(observed_indices):
        diff = observations[j] - ensemble[:, idx]
        log_likelihoods -= 0.5 * (diff ** 2) / obs_var

    # Normalize log-likelihoods for numerical stability
    log_likelihoods -= np.max(log_likelihoods)
    weights = np.exp(log_likelihoods)
    weights /= np.sum(weights)

    # Adaptive inflation: if effective sample size is too low, inflate
    n_eff = 1.0 / np.sum(weights ** 2)
    if n_eff < n_members * 0.3:
        # Inflate by increasing variance of unobserved components
        inflation_factor = 1.0 + 0.5 * (1.0 - n_eff / n_members)
        # Apply inflation to all components
        for i in range(n_members):
            ensemble[i] = ensemble[i] * inflation_factor

    # Importance resampling (systematic resampling)
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0
    u0 = np.random.uniform(0.0, 1.0 / n_members)
    u = u0 + np.arange(n_members) / n_members
    indices = np.searchsorted(cumulative_weights, u)
    indices = np.clip(indices, 0, n_members - 1)

    resampled = ensemble[indices]

    # KDE-based posterior update: blend resampled particles with a KDE-estimated mean
    # Compute bandwidth for KDE on observed components
    bandwidths = _bandwidth_selection(resampled, observed_indices)

    # Estimate posterior mean using KDE at observation locations
    # For each observed index, compute weighted KDE mean
    posterior_means = np.zeros(n_state)
    for j, idx in enumerate(observed_indices):
        vals = resampled[:, idx]
        bw = bandwidths[j]
        if bw < 1e-8:
            bw = 0.1
        # KDE mean: sum_k [K(obs_j - vals_k) * vals_k] / sum_k [K(obs_j - vals_k)]
        kernel_vals = np.exp(-0.5 * ((observations[j] - vals) / bw) ** 2)
        kernel_vals /= (bw * np.sqrt(2.0 * np.pi))
        if np.sum(kernel_vals) > 1e-15:
            posterior_means[idx] = np.sum(kernel_vals * vals) / np.sum(kernel_vals)
        else:
            posterior_means[idx] = observations[j]

    # For unobserved indices, use ensemble mean from resampled particles
    for idx in range(n_state):
        if idx not in observed_indices:
            posterior_means[idx] = np.mean(resampled[:, idx])

    # Blend: for observed indices, use posterior mean; for unobserved, keep particle values
    # But we need to maintain ensemble structure. Use a combination:
    # Updated particle = particle + alpha * (posterior_mean - particle) for observed
    # For unobserved, add small perturbation based on ensemble spread
    alpha = 0.7  # blending factor

    updated = resampled.copy()
    for j, idx in enumerate(observed_indices):
        updated[:, idx] = updated[:, idx] + alpha * (posterior_means[idx] - updated[:, idx])

    # Add small random perturbation to unobserved components to maintain diversity
    unobserved_indices = [i for i in range(n_state) if i not in observed_indices]
    for idx in unobserved_indices:
        spread = np.std(resampled[:, idx])
        if spread < 1e-8:
            spread = 0.5
        updated[:, idx] += np.random.normal(0, 0.1 * spread, size=n_members)

    # Clamp to valid range
    updated = np.clip(updated, -100.0, 100.0)

    # Ensure finite values
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)

    # Check deadline
    if time.time() > deadline:
        return updated

    return updated

# EVOLVE-BLOCK-END
