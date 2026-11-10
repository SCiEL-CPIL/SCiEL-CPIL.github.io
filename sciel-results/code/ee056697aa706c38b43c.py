"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    result[mask1] = (-5.0 / 4.0) * d[mask1]**5 / c**5 + 5.0 * d[mask1]**4 / c**4 - 10.0 * d[mask1]**3 / c**3 + 5.0 * d[mask1]
    # 0.5c < d <= 1.5c
    result[mask2] = (1.0 / 12.0) * d[mask2]**5 / c**5 - (5.0 / 4.0) * d[mask2]**4 / c**4 + (5.0 / 2.0) * d[mask2]**3 / c**3 - 5.0 * d[mask2]**2 / c**2 + (25.0 / 12.0) * d[mask2] / c + 1.0 / 2.0
    # 1.5c < d <= 2.5c
    result[mask3] = (-1.0 / 16.0) * d[mask3]**5 / c**5 + (1.0 / 4.0) * d[mask3]**4 / c**4 - (15.0 / 8.0) * d[mask3]**3 / c**3 + (5.0 / 2.0) * d[mask3]**2 / c**2 - (25.0 / 12.0) * d[mask3] / c + 105.0 / 16.0
    
    return result


def _localize_cov(cov, obs_idx, n, c=1.5):
    """Apply Gaspari-Cohn localization to observation covariance matrix."""
    m = len(obs_idx)
    if m <= 1:
        return cov
    
    # Compute distance matrix in circular space
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            d = abs(obs_idx[i] - obs_idx[j])
            dist[i, j] = min(d, n - d)
    
    loc = _gaspari_cohn(dist, c)
    return cov * loc


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
    
    # Localize the state-state covariance for observed variables
    # We need P_obs = P[obs_idx, obs_idx]
    P_obs = P[np.ix_(obs_idx, obs_idx)]
    
    # Apply localization to observation block
    P_obs_loc = _localize_cov(P_obs, obs_idx, n_state, c=1.5)
    
    # Cross-covariance between all states and observed states
    P_xo = P[np.ix_(range(n_state), obs_idx)]
    
    # Also localize cross-covariance
    # For each state i and observed state j, localize based on distance
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(n_state):
        for j in range(n_obs):
            d = abs(i - obs_idx[j])
            d = min(d, n_state - d)
            P_xo_loc[i, j] = P_xo[i, j] * _gaspari_cohn(np.array([d]), c=1.5)[0]
    
    # Kalman gain: K = P_xo_loc @ inv(P_obs_loc + R)
    # Use pseudo-inverse for numerical stability
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
    # Standard EnKF update: x_a = x_f + K @ (y - H(x_f))
    # For each member, we need to account for the nonlinear observation operator
    # Here H is linear (identity on observed indices), so:
    # x_a(i) = x_f(i) + K @ (y - H(x_f(i)))
    
    result = ensemble.copy()
    for m in range(n_members):
        Hx_m = result[m, obs_idx]
        innov_m = observations - Hx_m
        result[m] += K @ innov_m
    
    # Adaptive inflation to prevent filter collapse
    # Check if ensemble spread is too small
    ens_spread = np.std(result, axis=0)
    min_spread = 0.5
    
    # Inflate if any component has spread below threshold
    needs_inflation = np.any(ens_spread < min_spread)
    
    if needs_inflation:
        # Compute inflation factor
        # Target spread based on observation noise
        target_spread = np.sqrt(observation_variance)
        
        # Per-component inflation
        for i in range(n_state):
            if ens_spread[i] < min_spread:
                # Inflate this component
                inflation = max(1.0, target_spread / (ens_spread[i] + 1e-10))
                inflation = min(inflation, 2.0)  # Cap inflation
                mean_i = np.mean(result[:, i])
                result[:, i] = mean_i + inflation * (result[:, i] - mean_i)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
