"""Lorenz-96 assimilation using a localized ensemble Kalman filter with adaptive inflation and iterative scattering."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_step(x, F, dt=0.01):
    """
    One RK4 step for the Lorenz-96 system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    with periodic boundary conditions.
    """
    n = len(x)
    
    def f(x):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized ensemble Kalman filter analysis update with adaptive inflation
    and iterative multiple scattering for non-linear error structure.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Compute ensemble covariance
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Get observed state components from ensemble
    ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
    ens_obs_mean = np.mean(ens_obs, axis=0)
    
    # Compute innovation
    innovation = observations - ens_obs_mean
    
    # Apply rational localization (Hamill et al.) vectorized
    decay_length = 2.5
    indices = np.arange(n_state)
    dist_matrix = np.abs(indices[:, None] - observed_indices[None, :])
    dist_matrix = np.minimum(dist_matrix, n_state - dist_matrix)
    weights = 1.0 / (1.0 + (dist_matrix / decay_length) ** 2)
    
    # Iterative EnKF update: apply the update multiple times to capture non-linear structure
    n_scatter = 3
    relaxation = 0.92
    for scatter_iter in range(n_scatter):
        # Recompute ensemble statistics
        ens_mean = np.mean(ens, axis=0)
        anomalies = ens - ens_mean
        
        # Get observed state components
        ens_obs = ens[:, observed_indices]
        ens_obs_mean = np.mean(ens_obs, axis=0)
        
        # Compute innovation
        innovation = observations - ens_obs_mean
        
        # Compute innovation covariance S = H P H^T + R
        P_obs_obs = (anomalies[:, observed_indices].T @ anomalies[:, observed_indices]) / (n_members - 1)
        S = P_obs_obs + R
        S_reg = S + 1e-6 * np.eye(n_obs)
        
        # EnKF update: transform anomalies in observation space
        # H_anom = anomalies[:, observed_indices] (n_members, n_obs)
        H_anom = anomalies[:, observed_indices]
        
        # Compute the transformation matrix
        # We want: new_anom_obs = H_anom + (I - K H) * H_anom ... 
        # Standard EnKF: x_a = x_f + K (y - H x_f)
        # For anomalies: a_a = a_f + K (y - H x_f - H a_f) = a_f - K H a_f (since y - H x_f is constant)
        # So a_a = (I - K H) a_f
        # K H = P H^T S^{-1} H
        # In EnKF form: a_a = a_f - a_f_obs @ (S^{-1} H^T P) ... 
        # Simpler: use the standard EnKF formula
        # a_a = a_f - a_f_obs @ S^{-1} @ (a_f_obs^T @ a_f) / (n-1) ... no
        # Standard: a_a = a_f + K * (y - H x_f - H a_f) where K = P H^T S^{-1}
        # = a_f - K H a_f
        # K H = P H^T S^{-1} H
        # P H^T = (a_f^T a_f)/(n-1) but we need the cross-covariance
        # P H^T = (anomalies^T @ anomalies[:, observed_indices]) / (n-1)  (n_state, n_obs)
        PHt = (anomalies.T @ anomalies[:, observed_indices]) / (n_members - 1)
        
        try:
            K = np.linalg.solve(S_reg, PHt.T).T  # (n_state, n_obs)
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S_reg)
        
        # Apply relaxation
        K *= relaxation
        
        # Apply localization
        localized_K = K * weights  # (n_state, n_obs)
        
        # Apply update to each ensemble member
        member_innovations = observations[None, :] - ens_obs
        update = member_innovations @ localized_K.T
        ens = ens + update
        
        # Check deadline
        if _check_deadline():
            break
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    
    # Get observed state components after update
    ens_obs_new = ens[:, observed_indices]
    ens_obs_mean_new = np.mean(ens_obs_new, axis=0)
    
    # Compute the spread of observations in the ensemble
    obs_anoms = ens_obs_new - ens_obs_mean_new
    obs_var_ens = np.mean(obs_anoms ** 2, axis=0)  # (n_obs,)
    
    # Expected observation variance from ensemble + noise
    # The ensemble spread in observation space should match sqrt(obs_var_ens)
    # We want: obs_var_ens_inflated + obs_var_noise ~ target
    # Use a simple spread matching: inflate so that ensemble spread in obs space
    # is consistent with observation variance
    target_spread = np.sqrt(observation_variance)
    
    # Compute per-component inflation
    valid = obs_var_ens > 1e-10
    inflation_factors = np.ones(n_obs)
    if np.any(valid):
        inflation_factors[valid] = target_spread / np.sqrt(obs_var_ens[valid])
    
    # Use median for robustness, clip to reasonable range
    med_inflation = np.median(inflation_factors)
    inflation = np.clip(med_inflation, 0.85, 1.3)
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
