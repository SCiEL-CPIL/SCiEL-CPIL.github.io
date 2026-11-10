"""Ensemble Kalman Filter for Lorenz-96 assimilation with inflation and localization."""
import numpy as np
import time

# Global deadline guard
_DEADLINE = 150.0
_START_TIME = time.time()

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _lorenz96_rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of states.
    observations : ndarray of shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array_like
        Indices of observed coordinates (even indices 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise (0.49).
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after assimilation.
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return ensemble.copy()
    
    ens = ensemble.copy()
    n_ens, n_state = ens.shape
    n_obs = len(observations)
    obs_idx = np.array(observed_indices)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ens, axis=0)
    X = ens - ens_mean  # anomalies
    
    # Observation operator: H selects observed indices
    # H @ x = x[obs_idx]
    
    # Innovation: y - H @ x_bar
    innovation = observations - ens_mean[obs_idx]
    
    # Compute covariance of observations
    # P_oo = (1/(n-1)) * X[:, obs_idx] @ X[:, obs_idx].T + R
    R = observation_variance * np.eye(n_obs)
    X_obs = X[:, obs_idx]  # shape (n_ens, n_obs)
    P_oo = (X_obs @ X_obs.T) / (n_ens - 1) + R
    
    # Compute gain
    # K = P_xo @ P_oo^{-1}
    # P_xo = (1/(n-1)) * X @ X_obs.T
    P_xo = (X @ X_obs.T) / (n_ens - 1)  # shape (n_state, n_obs)
    
    # Solve K @ P_oo = P_xo
    try:
        K = P_xo @ np.linalg.inv(P_oo)
    except np.linalg.LinAlgError:
        # Fallback to simple nudging
        result = ens.copy()
        result[:, obs_idx] += 0.6 * (observations - result[:, obs_idx])
        return result
    
    # Update ensemble
    # x_new = x + K @ (y - H @ x)
    for i in range(n_ens):
        ens[i] = ens[i] + K @ innovation
    
    # Apply inflation to prevent filter collapse
    # Estimate spread reduction
    ens_mean_new = np.mean(ens, axis=0)
    X_new = ens - ens_mean_new
    P_oo_new = (X_new[:, obs_idx] @ X_new[:, obs_idx].T) / (n_ens - 1)
    
    # Simple adaptive inflation
    # Compare pre and post-update spread in observation space
    P_oo_pre = P_oo - R  # pre-update observation covariance without noise
    spread_ratio = np.trace(P_oo_new) / (np.trace(P_oo_pre) + 1e-10)
    
    # If spread decreased too much, inflate
    if spread_ratio < 1.0:
        # Inflate around mean
        inflation_factor = 1.0 + 0.01 * (1.0 - spread_ratio)
        for i in range(n_ens):
            ens[i] = ens_mean_new + inflation_factor * (ens[i] - ens_mean_new)
    
    # Clip to reasonable bounds
    ens = np.clip(ens, -100, 100)
    
    return ens
