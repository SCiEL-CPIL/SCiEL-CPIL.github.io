"""Improved ensemble assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_state = {"alpha": 1.0}
_MAX_INFLATION = 1.05
_MIN_INFLATION = 0.99
_INFLATION_RATE = 0.05  # how fast we adapt

def _lorenz96_step(x, F, dt=0.01, n_steps=5):
    """Propagate Lorenz-96 state forward using RK4 for n_steps of dt."""
    n = len(x)
    for _ in range(n_steps):
        def f(state):
            return np.array([
                (state[(i+1)%n] - state[(i-2)%n]) * state[(i-1)%n] - state[i] + F
                for i in range(n)
            ])
        k1 = f(x)
        k2 = f(x + 0.5*dt*k1)
        k3 = f(x + 0.5*dt*k2)
        k4 = f(x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization, adaptive inflation, and
    covariance inflation based on ensemble spread vs observation error.
    
    ensemble: (16, 20) array
    observations: (10,) array for even-indexed variables
    observed_indices: indices of observed variables (even: 0,2,4,...,18)
    observation_variance: scalar, variance of observation noise
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin
    
    N = ensemble.shape[0]  # ensemble size
    nvars = ensemble.shape[1]  # number of variables
    obs_idx = np.asarray(observed_indices)
    nobs = len(obs_idx)
    
    result = ensemble.copy()
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    ens_dev = ensemble - ens_mean[None, :]  # (N, nvars)
    
    # Sample covariance (unbiased)
    cov = (ens_dev.T @ ens_dev) / (N - 1.0)  # (nvars, nvars)
    
    # Observation covariance
    R = observation_variance * np.eye(nobs)
    
    # Localization: use exponential decay with e-folding scale
    # For Lorenz-96, correlations decay over a few grid points
    localization_scale = 3.0  # e-folding scale in grid indices
    
    # Localize the background covariance for the observed subset
    cov_obs_obs = cov[np.ix_(obs_idx, obs_idx)]
    
    # Apply localization
    loc_mask = np.zeros((nobs, nobs))
    for i in range(nobs):
        for j in range(nobs):
            # Ring distance on the 20-variable circle
            d = abs(obs_idx[i] - obs_idx[j])
            d = min(d, nvars - d)
            loc_mask[i, j] = np.exp(-d / localization_scale)
    
    cov_loc = cov_obs_obs * loc_mask
    
    # Ensure positive definiteness with Tikhonov regularization
    cov_loc += 1e-6 * np.eye(nobs)
    
    # Kalman gain computation
    # K = cov_bg_obs * (cov_obs_obs + R)^{-1}
    # cov_bg_obs = cov[:, obs_idx]
    cov_bg_obs = cov[:, obs_idx]  # (nvars, nobs)
    
    # Localize cross-covariance too
    cov_bg_obs_loc = np.zeros_like(cov_bg_obs)
    for j in range(nobs):
        for i in range(nvars):
            d = abs(i - obs_idx[j])
            d = min(d, nvars - d)
            cov_bg_obs_loc[i, j] = cov_bg_obs[i, j] * np.exp(-d / localization_scale)
    
    # Compute innovation
    innovation = observations - ens_mean[obs_idx]  # (nobs,)
    
    # Kalman gain
    S = cov_loc + R  # (nobs, nobs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = cov_bg_obs_loc @ S_inv  # (nvars, nobs)
    
    # Update ensemble mean
    updated_mean = ens_mean + K @ innovation
    
    # Update ensemble perturbations
    # For each member: x'_i = x_i + K @ (y - H(x_i))
    for m in range(N):
        x_obs = ensemble[m, obs_idx]
        innov_m = observations - x_obs
        result[m, :] = ensemble[m, :] + K @ innov_m
    
    # Adaptive variance inflation
    # Compare ensemble spread at observed points to observation error
    ens_spread_obs = np.std(result[:, obs_idx], axis=0, ddof=1)
    obs_std = np.sqrt(observation_variance)
    
    # Target spread should be roughly consistent with observation error
    # If spread is too small, inflate; if too large, deflate slightly
    ratio = np.mean(ens_spread_obs / obs_std) if obs_std > 0 else 1.0
    
    # Adapt inflation factor
    if ratio < 1.0:
        # Under-dispersed: increase inflation
        _inflation_state["alpha"] = min(
            _inflation_state["alpha"] * (1 + _INFLATION_RATE * (1.0 - ratio)),
            _MAX_INFLATION
        )
    elif ratio > 1.5:
        # Over-dispersed: decrease inflation
        _inflation_state["alpha"] = max(
            _inflation_state["alpha"] * (1 - _INFLATION_RATE * (ratio - 1.0)),
            _MIN_INFLATION
        )
    
    alpha = _inflation_state["alpha"]
    
    # Apply inflation around the updated mean
    updated_mean_final = np.mean(result, axis=0)
    for m in range(N):
        result[m, :] = updated_mean_final + alpha * (result[m, :] - updated_mean_final)
    
    # Final safety: clip to [-100, 100] and ensure finiteness
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
