"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Lorenz-96 RHS for a single state vector x of length 40 (20 vars)."""
    n = len(x) // 2
    x1 = x[:n]
    x2 = x[n:]
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x1[(i+1) % n] - x1[(i-2) % n]) * x1[(i-1) % n] - x1[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x) // 2
    x1 = x[:n]
    x2 = x[n:]
    k1 = _lorenz96_rhs(x, F)
    x_temp = x + 0.5 * dt * k1
    k2 = _lorenz96_rhs(x_temp, F)
    x_temp = x + 0.5 * dt * k2
    k3 = _lorenz96_rhs(x_temp, F)
    x_temp = x + dt * k3
    k4 = _lorenz96_rhs(x_temp, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _infer_F(ensemble):
    """Infer forcing parameter F from ensemble mean."""
    mean_x = np.mean(ensemble, axis=0)
    n = len(mean_x)
    # Estimate F from the balance: <dx_i/dt> ≈ 0 in steady state
    # F ≈ <x_i> - <(x_{i+1}-x_{i-2})*x_{i-1}>
    F_estimates = []
    for i in range(n):
        term = (mean_x[(i+1) % n] - mean_x[(i-2) % n]) * mean_x[(i-1) % n]
        F_estimates.append(mean_x[i] - term)
    return np.mean(F_estimates)

def _localize_cov(cov, observed_indices, all_indices, localization_width=3.0):
    """Apply Gaussian localization to covariance matrix for observed variables."""
    n = len(all_indices)
    m = len(observed_indices)
    # Build localization matrix for observed-observed block
    loc_mat = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            dist = abs(all_indices[observed_indices[i]] - all_indices[observed_indices[j]])
            # Minimum image distance for periodic system
            dist = min(dist, n - dist)
            loc_mat[i, j] = np.exp(-0.5 * (dist / localization_width) ** 2)
    return loc_mat

def _ensemble_kalman_filter(ensemble, observations, observed_indices, observation_variance, 
                            localization_width=2.5, inflation_rate=1.01):
    """
    Ensemble Kalman Filter update for Lorenz-96.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of observations at even indices
    observed_indices: indices that are observed (even: 0,2,4,...,18)
    observation_variance: scalar variance of observation noise
    """
    n_members, n_vars = ensemble.shape
    m_obs = len(observed_indices)
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(ensemble, axis=0)
    ens_anom = ensemble - ens_mean
    
    # Compute background covariance (16x16 -> use sample covariance)
    # B = ens_anom.T @ ens_anom / (n_members - 1)
    # But we work in anomaly space
    
    # Observation operator: H picks out observed_indices
    # H is (m_obs, n_vars)
    H = np.zeros((m_obs, n_vars))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation covariance
    R = observation_variance * np.eye(m_obs)
    
    # Background covariance projected to observation space
    # P_oo = H @ P @ H^T
    # P = ens_anom.T @ ens_anom / (n_members - 1)
    P_oo = np.zeros((m_obs, m_obs))
    for i in range(m_obs):
        for j in range(m_obs):
            P_oo[i, j] = np.mean(ens_anom[:, observed_indices[i]] * ens_anom[:, observed_indices[j]])
    
    # Apply localization to P_oo
    all_indices = list(range(n_vars))
    loc_mat = _localize_cov(P_oo, observed_indices, all_indices, localization_width)
    P_oo_localized = P_oo * loc_mat
    
    # Innovation
    innovation = observations - (H @ ens_mean)
    
    # Kalman gain: K = P @ H^T @ (H @ P @ H^T + R)^-1
    # In ensemble form: K = (ens_anom @ H_anom^T) @ (H @ P @ H^T + R)^-1
    # where H_anom = ens_anom @ H^T (m_obs x n_members)
    H_anom = ens_anom[:, observed_indices]  # (n_members, m_obs)
    
    # S = H @ P @ H^T + R (with localization)
    S = P_oo_localized + R
    
    # Inverse of S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain in ensemble form: K = (ens_anom @ H_anom^T @ S_inv)
    # K shape: (n_members, m_obs)
    K = H_anom @ S_inv  # (n_members, m_obs)
    
    # Update ensemble
    # x_new = x + K @ (y - H @ x)
    # For each member: x_new[:, i] = x[:, i] + K[:, i] @ (y - H @ x[:, i])
    # H @ x[:, i] = x[:, i, observed_indices]
    
    updated_ensemble = ensemble.copy()
    for i in range(n_members):
        obs_pred = ensemble[i, observed_indices]
        inn = observations - obs_pred
        updated_ensemble[i, observed_indices] += K[i, :] @ inn
    
    # Apply inflation to prevent filter collapse
    if inflation_rate > 1.0:
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation_rate * (updated_ensemble - updated_mean)
    
    return updated_ensemble

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 partial observation assimilation.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of observations at even indices
    observed_indices: array of observed indices (0, 2, 4, ..., 18)
    observation_variance: scalar variance of observation noise
    
    Returns:
    Updated ensemble with same (16, 20) shape
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Ensure inputs are numpy arrays
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_vars = ensemble.shape
    
    # Try ensemble Kalman filter
    try:
        updated = _ensemble_kalman_filter(
            ensemble, observations, observed_indices, observation_variance,
            localization_width=2.5, inflation_rate=1.02
        )
        
        # Ensure finite values
        if not np.all(np.isfinite(updated)):
            updated = ensemble.copy()
        
        # Clip to valid range
        updated = np.clip(updated, -100.0, 100.0)
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            # Fallback to simple nudging if too slow
            result = ensemble.copy()
            result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
            result = np.clip(result, -100.0, 100.0)
            return result
        
        return updated
        
    except Exception:
        # Fallback to simple nudging
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        result = np.clip(result, -100.0, 100.0)
        return result
# EVOLVE-BLOCK-END
