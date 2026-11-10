"""Lorenz-96 ensemble assimilation using Ensemble Kalman Filter with localization and inflation."""
import numpy as np
from scipy.linalg import solve

def _lorenz96_step(x, F, dt=0.01):
    """One RK4 step for Lorenz-96 system."""
    n = len(x)
    
    def f(state):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : np.ndarray, shape (16, 20)
        Ensemble of 16 members, each with 20 Lorenz-96 variables.
    observations : np.ndarray, shape (10,)
        Noisy observations of the observed variables (even indices).
    observed_indices : np.ndarray or list, shape (10,)
        Indices of observed variables.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : np.ndarray, shape (16, 20)
        Updated ensemble after analysis step.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Compute ensemble mean and covariance
    x_mean = np.mean(ensemble, axis=0)
    
    # Perturbed ensemble for covariance estimation
    x_pert = ensemble - x_mean
    
    # Observation operator: H selects observed components
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Localized EnKF update (sequential, one observation at a time for better conditioning)
    updated_ensemble = ensemble.copy()
    updated_mean = x_mean.copy()
    
    # Localization parameter (distance-based)
    # For 20-variable Lorenz-96, use a moderate localization radius
    loc_radius = 6.0
    
    for k in range(n_obs):
        obs_val = observations[k]
        obs_idx = observed_indices[k]
        
        # Compute ensemble mean and covariance for this observation
        obs_mean = np.mean(updated_ensemble[:, obs_idx])
        obs_cov = np.cov(updated_ensemble[:, observed_indices], rowvar=False)
        
        # For simplicity, use the full covariance but apply localization
        # Compute analysis gain using the full state covariance
        S = np.cov(updated_ensemble, rowvar=False)  # n_state x n_state
        S_obs = np.cov(updated_ensemble[:, observed_indices], rowvar=False)  # n_obs x n_obs
        
        # Cross-covariance between state and observations
        C_xo = np.cov(updated_ensemble, updated_ensemble[:, observed_indices], rowvar=False)  # n_state x n_obs
        
        # Innovation
        y_innov = observations - np.mean(updated_ensemble[:, observed_indices])
        
        # Analysis gain
        # K = C_xo @ inv(S_obs + R)
        # Use Cholesky or solve for numerical stability
        try:
            S_obs_R = S_obs + R
            # Solve C_xo @ S_obs_R^{-1} without explicit inverse
            K = np.linalg.solve(S_obs_R.T, C_xo.T).T  # n_state x n_obs
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            K = C_xo @ np.linalg.pinv(S_obs_R)
        
        # Apply localization to the gain
        # Zero out gains for variables far from the observed variable
        for j in range(n_state):
            dist = min(abs(j - obs_idx), n_state - abs(j - obs_idx))  # periodic distance
            if dist > loc_radius:
                K[j, k] = 0.0
        
        # Update ensemble
        updated_ensemble[:, :] += np.outer(updated_ensemble[:, :] - np.mean(updated_ensemble, axis=0), 
                                            np.zeros(n_state))  # placeholder, will do proper update
        
        # Proper update: x_a = x_f + K @ (y - H @ x_f)
        # For each member:
        for m in range(n_members):
            innov = observations - np.mean(updated_ensemble[:, observed_indices])
            updated_ensemble[m, :] += K @ innov
    
    # Adaptive inflation to prevent ensemble collapse
    # Compute spread-skill ratio and inflate if needed
    spread = np.std(updated_ensemble, axis=0)
    # Simple inflation: ensure spread doesn't go below a threshold
    min_spread = 0.5
    for i in range(n_state):
        if spread[i] < min_spread:
            inflation_factor = min_spread / max(spread[i], 1e-10)
            updated_ensemble[:, i] = np.mean(updated_ensemble[:, i]) + \
                                     inflation_factor * (updated_ensemble[:, i] - np.mean(updated_ensemble[:, i]))
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
