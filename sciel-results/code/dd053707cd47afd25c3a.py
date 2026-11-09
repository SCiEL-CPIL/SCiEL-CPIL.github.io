"""Lorenz-96 ensemble assimilation with local ensemble square-root filter (LESRF)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96(x, F):
    """Compute Lorenz-96 dynamics: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5 * dt * k1, F)
    k3 = _lorenz96(x + 0.5 * dt * k2, F)
    k4 = _lorenz96(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _rk4_propagate(x, F, dt, steps):
    """Propagate a single state through multiple RK4 steps."""
    for _ in range(steps):
        x = _rk4_step(x, F, dt)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LESRF-based ensemble analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Convert to float
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R = observation_variance
    
    # Observation error variance (scalar)
    R_val = float(R) if np.isscalar(R) else float(np.mean(R))
    
    # Precompute localization weights using a sharper Gaussian
    # Use a correlation length tuned for Lorenz-96 spatial correlation decay
    corr_length = 3.5
    
    # Compute localization weights for each observed variable to each state variable
    # For each observed index, compute weights for all 20 variables
    loc_weights = np.zeros((n_obs, n_vars))
    for j, obs_i in enumerate(obs_idx):
        for i in range(n_vars):
            # Periodic distance on the ring
            dist = abs(i - obs_i)
            dist = min(dist, n_vars - dist)
            # Use a sharper Gaussian for better localization
            loc_weights[j, i] = np.exp(-0.5 * (dist / corr_length) ** 2)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Adaptive inflation: check if ensemble spread is consistent with observations
    # Compute ensemble spread at observed locations
    ens_spread_obs = np.zeros(n_obs)
    for j in range(n_obs):
        h_ens = ens[:, obs_idx[j]]
        ens_spread_obs[j] = np.var(h_ens, ddof=1)
    
    # Compare ensemble spread to observation variance
    # If spread is too small relative to obs error, inflate more
    spread_ratio = np.mean(ens_spread_obs) / R_val if R_val > 0 else 1.0
    
    # Adaptive inflation: base 1.02, increase if spread too small, decrease if too large
    if spread_ratio < 0.5:
        inflation = 1.08
    elif spread_ratio < 1.0:
        inflation = 1.04
    elif spread_ratio < 2.0:
        inflation = 1.01
    else:
        inflation = 1.00
    
    # Apply inflation to the ensemble
    ens_centered = ens - ens_mean
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    
    # Compute ensemble covariance (anomalies)
    anomalies = ens - ens_mean  # shape (n_members, n_vars)
    
    # Precompute the ensemble size factor
    alpha = n_members / (n_members - 1.0)
    
    # Process observations sequentially
    for j in range(n_obs):
        # Check time
        if time.time() > deadline:
            break
        
        obs_j = obs[j]
        obs_idx_j = obs_idx[j]
        
        # Localization weights for this observation
        w = loc_weights[j, :]  # shape (n_vars,)
        
        # If the weight for the observed variable itself is too small, skip
        if w[obs_idx_j] < 0.01:
            continue
        
        # Compute the observation operator applied to ensemble
        # H(x) = x[obs_idx_j]
        H_ens = ens[:, obs_idx_j]  # shape (n_members,)
        H_mean = np.mean(H_ens)
        H_anom = H_ens - H_mean
        
        # Innovation: observation - ensemble mean observation
        innovation = obs_j - H_mean
        
        # Compute the variance of the observation in the ensemble
        # S = alpha * var(H_ens) + R
        h_var = alpha * np.dot(H_anom, H_anom) / (n_members - 1.0)
        S = h_var + R_val
        
        # Gain factor
        K = h_var / S
        
        # Update the ensemble using the square-root filter
        # For the observed variable and nearby variables
        for i in range(n_vars):
            if w[i] < 0.01:
                continue
            
            # Compute the cross-covariance between variable i and the observation
            x_anom = anomalies[:, i]
            cross_cov = alpha * np.dot(x_anom, H_anom) / (n_members - 1.0)
            
            # Apply localization
            cross_cov_loc = cross_cov * w[i]
            
            # Update
            ens[:, i] += (cross_cov_loc / S) * (obs - H_ens)
        
        # Recompute anomalies after update
        ens_mean = np.mean(ens, axis=0)
        anomalies = ens - ens_mean
    
    # Step 4: Final inflation and clipping
    # Apply a small multiplicative inflation to counteract filter divergence
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Check ensemble spread after update for final inflation
    final_spread = np.mean(np.var(ens, axis=0, ddof=1))
    if final_spread < R_val * 0.8:
        final_inflation = 1.03
    elif final_spread < R_val:
        final_inflation = 1.015
    else:
        final_inflation = 1.005
    
    ens = ens_mean + final_inflation * ens_centered
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finite values
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    # Check time
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
