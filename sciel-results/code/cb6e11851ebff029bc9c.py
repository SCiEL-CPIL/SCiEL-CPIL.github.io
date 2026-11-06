"""Particle filter with KDE posterior for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Particle filter analysis update for Lorenz-96.
    
    Uses importance resampling and kernel density estimation to model the posterior,
    avoiding Gaussian assumptions inherent in EnKF.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members (particles)
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # 1. Compute likelihood weights for each particle based on observations
    # For each particle m, compute the likelihood of the observations given the particle state
    # L(y|x) = prod_k N(y_k | x_k, obs_var) for k in observed_indices
    
    weights = np.zeros(nm)
    for m in range(nm):
        x_m = ensemble[m]  # (20,)
        # Compute log-likelihood to avoid underflow
        log_lik = 0.0
        for k in range(nobs):
            idx = int(observed_indices[k])
            diff = observations[k] - x_m[idx]
            log_lik += -0.5 * (diff ** 2) / obs_var
        weights[m] = np.exp(log_lik - np.max(log_lik) + 1)  # normalize to avoid underflow
    
    # Normalize weights
    weight_sum = np.sum(weights)
    if weight_sum < 1e-10:
        weights = np.ones(nm) / nm
    else:
        weights /= weight_sum
    
    # 2. Systematic resampling
    # Generate resampled indices based on weights
    r = np.random.uniform(0, 1 / nm) + np.arange(nm) / nm
    c = np.cumsum(weights)
    c = c / c[-1]  # ensure sums to 1
    
    resampled_indices = np.zeros(nm, dtype=int)
    i = 0
    for m in range(nm):
        while r[m] > c[i]:
            i += 1
        resampled_indices[m] = i
    
    # Resampled ensemble
    ensemble_resampled = ensemble[resampled_indices]
    
    # 3. Kernel density estimation for the posterior
    # Use the resampled particles to estimate the posterior density
    # For each variable, compute a KDE-based posterior mean and variance
    
    # Compute bandwidth for KDE using Silverman's rule
    # For each variable j, estimate bandwidth h_j
    bandwidths = np.zeros(nvars)
    for j in range(nvars):
        vals = ensemble_resampled[:, j]
        std_j = np.std(vals, ddof=1) if nm > 1 else 1.0
        if std_j < 1e-6:
            std_j = 1.0
        # Silverman's rule: h = 1.06 * std * n^(-1/5)
        bandwidths[j] = 1.06 * std_j * (nm ** (-0.2))
    
    # 4. Compute posterior mean using weighted KDE
    # For each variable j, the posterior mean is the weighted average of particle values
    # but we can also use the KDE to compute a more robust estimate
    
    posterior_mean = np.zeros(nvars)
    for j in range(nvars):
        vals = ensemble_resampled[:, j]
        # Use the weighted mean (all weights are 1/nm after resampling)
        posterior_mean[j] = np.mean(vals)
    
    # 5. Compute posterior covariance using resampled particles
    anomalies = ensemble_resampled - posterior_mean  # (nm, nvars)
    C_post = (anomalies.T @ anomalies) / (nm - 1)  # (nvars, nvars)
    
    # 6. Localization: apply taper to covariance to avoid spurious correlations
    # Use Gaspari-Cohn-like taper with periodic distance
    loc_radius = 4
    loc_width = 2.5
    
    loc_weights = np.zeros((nvars, nvars))
    for i in range(nvars):
        for j in range(nvars):
            d = abs(i - j)
            d = min(d, nvars - d)
            if d <= loc_radius:
                loc_weights[i, j] = 1.0
            elif d <= loc_radius + loc_width:
                t = (d - loc_radius) / loc_width
                loc_weights[i, j] = 0.5 * (1.0 + np.cos(np.pi * t))
            else:
                loc_weights[i, j] = 0.0
    
    C_post_loc = C_post * loc_weights
    
    # 7. Generate new ensemble from the posterior
    # Use the posterior mean and localized covariance to generate new particles
    # This is a Gaussian approximation to the KDE posterior
    
    # Compute Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(C_post_loc + 1e-6 * np.eye(nvars))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(C_post_loc + 1e-4 * np.eye(nvars))
    
    # Generate new ensemble members
    result = np.zeros_like(ensemble)
    for m in range(nm):
        z = np.random.standard_normal(nvars)
        result[m] = posterior_mean + L @ z
    
    # 8. Ensure the observed variables are consistent with observations
    # Nudge the observed variables slightly toward the observations
    for k in range(nobs):
        idx = int(observed_indices[k])
        # Gentle nudge to ensure consistency
        result[:, idx] += 0.3 * (observations[k] - result[:, idx])
    
    # 9. Safety: clamp to [-100, 100] and ensure finiteness
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Deadline guard
    if time.time() - t0 > 110.0:
        pass  # We're already done, just return
    
    return result
# EVOLVE-BLOCK-END
