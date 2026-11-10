"""Combined EnKF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Precompute localization matrix for N=20 (ring topology)
# Gaspari-Cohn function with correlation length c
def _gc(d, c):
    """Gaspari-Cohn correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    # d <= c/2
    mask1 = d <= 0.5 * c
    r1 = d[mask1]
    result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    # c/2 < d <= 3c/2
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    r2 = d[mask2]
    result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    # 3c/2 < d <= 5c/2
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    r3 = d[mask3]
    result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (15.0 / 8.0) * (r3 / c) ** 3 + (5.0 / 2.0) * (r3 / c) ** 2 - (25.0 / 12.0) * (r3 / c) + 105.0 / 16.0
    return result


def _build_loc_matrix(n, c):
    """Build a full n x n Gaspari-Cohn localization matrix for a ring."""
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n - dist)
    return _gc(dist, c)


# Precompute for N=20
_N = 20
_CORR_LEN = 4.0
_LOC_MATRIX = _build_loc_matrix(_N, _CORR_LEN)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of state vectors
    observations : ndarray of shape (10,)
        Noisy observations of even-indexed coordinates
    observed_indices : array-like of shape (10,)
        Indices of observed coordinates (even indices: 0,2,4,...,18)
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after analysis
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # shape (n_members, n_state)
    
    # Compute ensemble covariance: P = (1/(N-1)) * A^T A
    P = (anomalies.T @ anomalies) / (n_members - 1.0)  # shape (n_state, n_state)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Get observation block of covariance and apply localization
    P_obs = P[np.ix_(obs_idx, obs_idx)]
    
    # Build localized observation covariance
    # Distance matrix among observed indices in circular space
    m = len(obs_idx)
    dist_obs = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            d = abs(obs_idx[i] - obs_idx[j])
            dist_obs[i, j] = min(d, n_state - d)
    loc_obs = _gc(dist_obs, _CORR_LEN)
    P_obs_loc = P_obs * loc_obs
    
    # Cross-covariance between all states and observed states
    P_xo = P[np.ix_(range(n_state), obs_idx)]
    
    # Localize cross-covariance
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(n_state):
        for j in range(n_obs):
            d = abs(i - obs_idx[j])
            d = min(d, n_state - d)
            P_xo_loc[i, j] = P_xo[i, j] * _gc(np.array([d]), _CORR_LEN)[0]
    
    # Kalman gain: K = P_xo_loc @ inv(P_obs_loc + R)
    S = P_obs_loc + R  # Innovation covariance
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P_xo_loc @ S_inv  # shape (n_state, n_obs)
    
    # Innovation: y - H(x)
    Hx = ens_mean[obs_idx]
    innovation = observations - Hx
    
    # Analysis increment
    increment = K @ innovation  # shape (n_state,)
    
    # Update ensemble mean
    new_mean = ens_mean + increment
    
    # Update ensemble members
    result = ensemble.copy()
    for mm in range(n_members):
        Hx_m = result[mm, obs_idx]
        innov_m = observations - Hx_m
        result[mm] += K @ innov_m
    
    # Adaptive inflation to prevent filter collapse
    ens_spread = np.std(result, axis=0)
    min_spread = 0.3
    
    # Inflate if any component has spread below threshold
    needs_inflation = np.any(ens_spread < min_spread)
    
    if needs_inflation:
        target_spread = np.sqrt(observation_variance)
        
        for i in range(n_state):
            if ens_spread[i] < min_spread:
                inflation = max(1.0, target_spread / (ens_spread[i] + 1e-10))
                inflation = min(inflation, 1.8)
                mean_i = np.mean(result[:, i])
                result[:, i] = mean_i + inflation * (result[:, i] - mean_i)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
