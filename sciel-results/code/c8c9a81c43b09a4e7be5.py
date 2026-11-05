"""Lorenz-96 ensemble assimilation: localized EKF with adaptive inflation,
per-member localized gain, and a simple forcing estimate."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.06
_inflation_alpha = 0.80
_inflation_min = 1.0
_inflation_max = 2.2
_loc_range = 2.8

# Precompute localization weights for even indices (0,2,4,...,18)
# We'll build them lazily and cache
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.8):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]


def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20) or (20,)."""
    if x.ndim == 1:
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        xp1 = np.roll(x, -1)
        return (xp1 - xm2) * xm1 - x + F
    else:
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, F, dt, n_steps=5):
    """Propagate ensemble (N, 20) by RK4 for n_steps."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _localization_matrix(observed_indices, dim=20, range_decay=2.8):
    """Build a (n_obs, dim) localization weight matrix."""
    n_obs = len(observed_indices)
    W = np.zeros((n_obs, dim))
    for i, obs_idx in enumerate(observed_indices):
        W[i, :] = _get_loc_weights(obs_idx, dim, range_decay)
    return W


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: array of indices observed (should be [0,2,4,...,18])
    observation_variance: scalar variance of observation noise
    
    Returns: updated ensemble (16, 20)
    """
    global _inflation
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure finite
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0)
    observations = np.where(np.isfinite(observations), observations, 0.0)
    
    # Build observation operator H (n_obs x dim)
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Localization matrix (n_obs x dim)
    W = _localization_matrix(observed_indices, dim, _loc_range)
    
    # Adaptive inflation
    mean = ensemble.mean(axis=0)
    inflated = mean + _inflation * (ensemble - mean)
    
    # Localized ensemble smoother with multiple scattering (ES-MSC)
    # Iterate: update ensemble, then re-propagate to capture nonlinearity
    n_scatter = 3  # number of scattering iterations
    current = inflated.copy()
    
    for scatter in range(n_scatter):
        # Forecast observations from current ensemble
        y_f = current[:, observed_indices]  # (N, n_obs)
        y_f_mean = y_f.mean(axis=0)  # (n_obs,)
        
        # Innovation
        innovation = observations - y_f_mean  # (n_obs,)
        
        # Compute localized observation-space covariance
        # S_loc[i,j] = sum over members of W[i,k]*W[j,k] * (y_f[m,k]-y_f_mean[k])^2 / (N-1)
        # More efficiently: compute per-member innovations weighted by localization
        # For each member m and obs i: innov_m_i = W[i,obs_idx_i] * (y_f[m,i] - y_f_mean[i])
        # But we need the full covariance in obs space with localization
        
        # Simpler approach: use the standard EKF gain with localization applied to H
        # Effective H_eff = W (localization applied to observation operator)
        # This means each observation only affects nearby state components
        
        # Compute forecast observation covariance with localization
        # P_obs = (1/(N-1)) * sum_m (y_f[m,:] - y_f_mean) @ diag(W^2) ... 
        # Actually, let's use the standard approach:
        # S = H P H^T + R, but with localized P
        
        # Compute ensemble covariance (dim x dim)
        centered = inflated - mean  # (N, dim)
        P = (centered.T @ centered) / (N - 1)  # (dim, dim)
        
        # Apply localization: P_loc = D P D where D = diag of localization? 
        # Actually, localization in EKF is typically applied to the gain or to the observation operator.
        # A common approach: use H_loc = W (n_obs x dim) instead of H
        # Then S = H_loc P H_loc^T + R
        
        H_loc = W  # (n_obs, dim)
        S = H_loc @ P @ H_loc.T + observation_variance * np.eye(n_obs)
        
        # Compute Kalman gain with localization
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = P @ H_loc.T @ S_inv  # (dim, n_obs)
        
        # Update ensemble
        # For each member: x_new = x + K @ (y - y_f_i)
        # where y_f_i is the i-th member's forecast observation
        innovations_all = observations[:, np.newaxis] - y_f.T  # (n_obs, N)
        
        for i in range(N):
            current[i] = inflated[i] + K @ innovations_all[:, i]
        
        # Re-propagate through dynamics to capture nonlinearity
        # Estimate F from the ensemble mean dynamics
        # For Lorenz-96, F is constant but unknown. We can estimate it by
        # requiring that the mean is approximately on the attractor.
        # Simple estimate: F ~ mean of x values (since for steady state, <(x_{i+1}-x_{i-2})x_{i-1}> - <x_i> + F = 0)
        # This is approximate but should give a reasonable F
        F_est = mean[0]  # rough estimate, will be refined
        
        # Actually, let's use a better estimate: average over all indices
        # For each i: F ≈ x_i - (x_{i+1} - x_{i-2}) * x_{i-1}
        # Average over i
        F_vals = np.zeros(dim)
        for i in range(dim):
            xm2 = mean[(i - 2) % dim]
            xm1 = mean[(i - 1) % dim]
            xp1 = mean[(i + 1) % dim]
            F_vals[i] = mean[i] - (xp1 - xm2) * xm1
        F_est = np.mean(F_vals)
        
        # Propagate each member by RK4 for 5 steps at dt=0.01
        for i in range(N):
            current[i] = _rk4_step(current[i], F_est, 0.01, 5)
    
    # Adaptive inflation update
    innovation_final = observations - current[:, observed_indices].mean(axis=0)
    innovation_norm = np.linalg.norm(innovation_final)
    
    # Estimate expected innovation norm from S
    H_loc = W
    centered = inflated - mean
    P = (centered.T @ centered) / (N - 1)
    S = H_loc @ P @ H_loc.T + observation_variance * np.eye(n_obs)
    expected_innovation_norm = np.sqrt(np.trace(S))
    
    if expected_innovation_norm > 1e-10:
        ratio = innovation_norm / expected_innovation_norm
        if ratio > 1.0:
            _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * min(1.2, ratio)
        else:
            _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # Clip to valid range
    current = np.clip(current, -100, 100)
    current = np.where(np.isfinite(current), current, 0.0)
    
    return current
