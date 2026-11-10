"""Ensemble Kalman Filter with Gaspari-Cohn localization for Lorenz-96 assimilation."""
import numpy as np
import time

# Precomputed constants
_N = 20  # state dimension
_N_OBS = 10  # number of observed variables

def _gaspari_cohn(d, c):
    """Compute Gaspari-Cohn correlation function for distance d and correlation length c."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=np.float64)
    
    # Piece 1: d <= c/2
    mask1 = d <= 0.5 * c
    if np.any(mask1):
        r = d[mask1] / c
        result[mask1] = (-5/4) * r**5 + 5 * r**4 - 10 * r**3 + 5 * r
    
    # Piece 2: c/2 < d <= 3c/2
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    if np.any(mask2):
        r = d[mask2] / c
        result[mask2] = (1/12) * r**5 - (5/4) * r**4 + (5/2) * r**3 - 5 * r**2 + (25/12) * r + 0.5
    
    # Piece 3: 3c/2 < d <= 5c/2
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    if np.any(mask3):
        r = d[mask3] / c
        result[mask3] = (-1/16) * r**5 + (1/4) * r**4 - (5/12) * r**3 + (5/12) * r**2 + (25/12) * r - 25/12
    
    return result

def _build_localization_matrix(obs_indices, c):
    """Build the localization matrix for the observed indices on a ring of size _N."""
    n_obs = len(obs_indices)
    L = np.ones((n_obs, n_obs), dtype=np.float64)
    for i in range(n_obs):
        for j in range(n_obs):
            if i == j:
                L[i, j] = 1.0
            else:
                # Ring distance
                d = abs(obs_indices[i] - obs_indices[j])
                d = min(d, _N - d)
                L[i, j] = _gaspari_cohn(d, c)
    return L

def _lorenz96_step(x, F, dt, n_steps=5):
    """Propagate Lorenz-96 state using RK4 for n_steps steps."""
    n = len(x)
    for _ in range(n_steps):
        k1 = _lorenz96_rhs(x, F, n)
        k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F, n)
        k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F, n)
        k4 = _lorenz96_rhs(x + dt * k3, F, n)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

def _lorenz96_rhs(x, F, n):
    """Compute Lorenz-96 right-hand side."""
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis step with Gaspari-Cohn localization.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of states.
    observations : ndarray of shape (10,)
        Noisy observations of the even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after analysis.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin
    
    ens = ensemble.copy()
    n_ens, n_state = ens.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensemble mean and anomaly
    x_mean = np.mean(ens, axis=0)
    X_a = ens - x_mean  # Anomaly matrix, shape (n_ens, n_state)
    
    # Observation operator: extract observed components
    H_Xa = X_a[:, obs_idx]  # shape (n_ens, n_obs)
    H_x = x_mean[obs_idx]   # shape (n_obs,)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    y_innov = observations - H_x  # shape (n_obs,)
    
    # Compute ensemble covariance of observations
    # S = H P H^T + R
    # P is estimated from ensemble: P = (1/(n_ens-1)) * X_a X_a^T
    # S = (1/(n_ens-1)) * H_Xa @ H_Xa^T + R
    S = (H_Xa @ H_Xa.T) / (n_ens - 1) + R
    
    # Localization: apply Gaspari-Cohn to the observation covariance
    # Use a moderate correlation length
    c = 3.0
    L_matrix = _build_localization_matrix(obs_idx, c)
    
    # Localized S
    S_loc = L_matrix * S  # Element-wise multiplication for localization
    
    # Invert localized S
    try:
        S_inv = np.linalg.inv(S_loc)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_loc)
    
    # Compute Kalman gain
    # K = P H^T S^{-1}
    # P H^T = (1/(n_ens-1)) * X_a @ H_Xa.T
    PHt = (X_a @ H_Xa.T) / (n_ens - 1)
    K = PHt @ S_inv  # shape (n_state, n_obs)
    
    # Update ensemble
    # x_a_new = x_a + K @ (y - H x_mean)
    K_innov = K @ y_innov  # shape (n_state,)
    ens = ens + np.outer(np.ones(n_ens), K_innov)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found
        ens = np.clip(ens, -100, 100)
        return np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Simple variance inflation to prevent filter collapse
    x_mean_new = np.mean(ens, axis=0)
    X_a_new = ens - x_mean_new
    P_new = (X_a_new @ X_a_new.T) / (n_ens - 1)
    
    # Inflation factor
    inflation = 1.01
    ens = x_mean_new + inflation * X_a_new
    
    # Ensure finite values and within bounds
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    ens = np.clip(ens, -100, 100)
    
    return ens
