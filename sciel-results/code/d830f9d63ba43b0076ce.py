"""Local ETKF-style analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a local ensemble transform Kalman filter (ETKF) analysis update.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of state vectors (16 members, 20 variables).
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed variables.
    observed_indices : array-like
        Indices of observed variables (even indices: 0,2,4,...,18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after analysis.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 20
    Nens = ensemble.shape[0]  # 16
    Nobs = len(observed_indices)  # 10
    obs_idx = np.asarray(observed_indices, dtype=int)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)  # (20,)
    anomalies = ensemble - mean  # (16, 20)
    
    # Estimate ensemble covariance
    cov = (anomalies.T @ anomalies) / (Nens - 1)  # (20, 20)
    
    # Observation operator: H selects observed indices
    H = np.zeros((Nobs, N))
    for i, idx in enumerate(obs_idx):
        H[i, int(idx)] = 1.0
    
    # Observation covariance
    R = np.eye(Nobs) * observation_variance  # (10, 10)
    
    # Innovation (observation minus ensemble mean at observed locations)
    innovation = observations - H @ mean  # (10,)
    
    # --- Localization using Gaspari-Cohn ---
    # Precompute localization matrix for the ring topology
    # Correlation length L ~ 3 (in index units)
    L = 3.0
    
    def _gc(d, L):
        """Gaspari-Cohn 5th-order correlation function."""
        d = np.abs(d)
        r = d / L
        result = np.zeros_like(d)
        mask1 = r <= 0.5
        mask2 = (r > 0.5) & (r <= 1.5)
        mask3 = (r > 1.5) & (r <= 2.5)
        
        r1 = r[mask1]
        if len(r1) > 0:
            result[mask1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
        
        r2 = r[mask2]
        if len(r2) > 0:
            result[mask2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
        
        r3 = r[mask3]
        if len(r3) > 0:
            result[mask3] = (-1.0/16.0)*r3**5 + (1.0/4.0)*r3**4 - (15.0/8.0)*r3**3 + (5.0/2.0)*r3**2 - (25.0/12.0)*r3 + 105.0/16.0
        
        return result
    
    # Build localization matrix for state-space (20x20)
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)  # Circular distance
            loc_matrix[i, j] = _gc(np.array([d]), L)[0]
    
    # Localized covariance
    cov_loc = cov * loc_matrix  # Element-wise localization
    
    # Localize cross-covariance P_xo
    # P_xo[i, k] = cov[i, obs_idx[k]]
    P_xo = cov[:, obs_idx]  # (20, 10)
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(N):
        for k in range(Nobs):
            d = abs(i - obs_idx[k])
            d = min(d, N - d)
            P_xo_loc[i, k] = P_xo[i, k] * _gc(np.array([d]), L)[0]
    
    # Localize observation-space covariance P_obs
    P_obs = cov[np.ix_(obs_idx, obs_idx)]  # (10, 10)
    P_obs_loc = np.zeros_like(P_obs)
    for i in range(Nobs):
        for j in range(Nobs):
            d = abs(obs_idx[i] - obs_idx[j])
            d = min(d, N - d)
            P_obs_loc[i, j] = P_obs[i, j] * _gc(np.array([d]), L)[0]
    
    # Innovation covariance with localization
    S = P_obs_loc + R  # (10, 10)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Localized Kalman gain
    K = P_xo_loc @ S_inv  # (20, 10)
    
    # Analysis update for mean
    mean_analysis = mean + K @ innovation  # (20,)
    
    # --- Ensemble update using localized gain ---
    # For each member: x_a(m) = x_f(m) + K @ (y - H(x_f(m)))
    updated_ensemble = ensemble.copy()
    for m in range(Nens):
        Hx_m = updated_ensemble[m, obs_idx]
        innov_m = observations - Hx_m
        updated_ensemble[m] += K @ innov_m
    
    # --- Adaptive inflation to prevent ensemble collapse ---
    # Compute spread in observed space
    ens_obs = updated_ensemble[:, obs_idx]
    ens_obs_mean = np.mean(ens_obs, axis=0)
    ens_obs_anom = ens_obs - ens_obs_mean
    actual_spread = np.mean(np.var(ens_obs_anom, axis=0))
    
    # Expected spread should be roughly observation_variance + model_error
    # If actual spread is too small, inflate
    expected_spread = observation_variance * 0.8  # Conservative estimate
    
    if actual_spread < expected_spread * 0.4 and actual_spread > 1e-10:
        inflation_factor = np.sqrt(expected_spread / actual_spread)
        inflation_factor = min(inflation_factor, 2.5)  # Cap inflation
        # Inflate around the analysis mean
        updated_ensemble = mean_analysis[None, :] + (updated_ensemble - mean_analysis[None, :]) * inflation_factor
    
    # Per-component inflation for any components with very low spread
    ens_spread = np.std(updated_ensemble, axis=0)
    min_spread = 0.3
    for i in range(N):
        if ens_spread[i] < min_spread:
            target_spread = np.sqrt(observation_variance) * 0.7
            inflation = max(1.0, target_spread / (ens_spread[i] + 1e-10))
            inflation = min(inflation, 2.0)
            mean_i = np.mean(updated_ensemble[:, i])
            updated_ensemble[:, i] = mean_i + inflation * (updated_ensemble[:, i] - mean_i)
    
    # Ensure output is finite and within bounds
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return updated_ensemble
    
    return updated_ensemble
