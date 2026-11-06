"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Module-level state for adaptive inflation and timing
_inflation_state = {}
_start_time = None

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
    - Localized cross-covariance (exponential decay)
    - Adaptive multiplicative inflation (per-variable)
    - Deadlines guard
    """
    M, N = ensemble.shape
    obs_var = observation_variance
    obs_idx = observed_indices
    
    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        # Return a safe fallback: nudge toward observations
        result = ensemble.copy()
        result[:, obs_idx] += 0.5 * (observations - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result
    
    # Copy ensemble
    result = ensemble.copy()
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean  # (M, N)
    
    # --- Compute background covariance (M x N) ---
    # We'll work with the ensemble anomalies directly
    
    # --- Localization: exponential decay of cross-covariance ---
    # For each observed variable j and all variables i, weight by exp(-|i-j|/L)
    # Use a localization length scale L. For N=20, L ~ 5-8 works well.
    L_loc = 6.0  # localization length scale
    
    # --- Observation innovation ---
    # observations shape: (M, len(obs_idx)) presumably
    # But wait - observations are of the TRUE state with noise, one per member?
    # Actually in ensemble assimilation, observations are typically a single vector
    # or replicated. Let's check: observations shape should be (M, len(obs_idx)) 
    # if each member has its own observation, or (len(obs_idx),) if shared.
    # Given the interface says observations, and the seed uses:
    #   result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
    # This implies observations has shape (M, len(obs_idx)) to broadcast with result[:, obs_idx]
    
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 1:
        obs_arr = np.tile(obs_arr, (M, 1))
    
    # Innovation: y - H(x) for observed components
    H_ens = result[:, obs_idx]  # (M, n_obs)
    innovation = obs_arr - H_ens  # (M, n_obs)
    
    # --- Compute Kalman gain with localization ---
    # For each observed variable j, we compute the gain for all variables i
    # Covariance: C = (1/(M-1)) * anomalies^T @ anomalies  (N x N)
    # But we apply localization to the cross-covariance
    
    n_obs = len(obs_idx)
    
    # Anomalies for observed variables: (M, n_obs)
    anom_obs = anomalies[:, obs_idx]  # (M, n_obs)
    
    # Compute the Kalman update
    # Standard EnKF: 
    #   K = C_bg @ H^T @ (H @ C_bg @ H^T + R)^{-1}
    # But with M=16, we can use the ensemble form:
    #   For each member m, x_m^a = x_m^b + K @ (y_m - H x_m^b)
    #   where K is computed from ensemble statistics
    
    # Using the "square root" or direct ensemble approach:
    # Let Y = anom_obs  (M x n_obs)
    # Let X = anomalies  (M x N)
    # 
    # The update in ensemble form:
    #   x_m^a = x_m^b + (1/(M-1)) * X^T @ Y @ (Y^T @ Y/(M-1) + R)^{-1} @ (y_m - H x_m^b)
    # 
    # But we need localization on X^T @ Y (the cross-covariance)
    
    # Compute cross-covariance with localization
    # X^T @ Y is (N x n_obs)
    # For each pair (i, j), weight by exp(-dist(i, obs_idx[j]) / L_loc)
    
    # Build localization matrix
    # dist[i, j] = min(|i - obs_idx[j]|, N - |i - obs_idx[j]|) due to periodicity
    loc_weights = np.zeros((N, n_obs))
    for j_idx, j in enumerate(obs_idx):
        for i in range(N):
            d = abs(i - j)
            d = min(d, N - d)  # periodic distance
            loc_weights[i, j_idx] = np.exp(-d / L_loc)
    
    # Cross-covariance: (N x n_obs) = (1/(M-1)) * anomalies^T @ anom_obs
    cross_cov = (anomalies.T @ anom_obs) / (M - 1)  # (N, n_obs)
    
    # Apply localization
    cross_cov_loc = cross_cov * loc_weights
    
    # Innovation covariance: (n_obs x n_obs) = (1/(M-1)) * anom_obs^T @ anom_obs + R
    innov_cov = (anom_obs.T @ anom_obs) / (M - 1) + obs_var * np.eye(n_obs)
    
    # Invert innovation covariance (small matrix, n_obs=10)
    try:
        innov_cov_inv = np.linalg.inv(innov_cov)
    except np.linalg.LinAlgError:
        innov_cov_inv = np.linalg.pinv(innov_cov)
    
    # Kalman gain: K = cross_cov_loc @ innov_cov_inv  (N x n_obs)
    K = cross_cov_loc @ innov_cov_inv
    
    # Apply update: result[:, i] += sum_j K[i,j] * innovation[:,j]
    # result_new = result + anomalies @ ... 
    # Actually: x_m^a = x_m^b + K @ (y_m - H x_m^b)
    # For all members: result += K @ innovation.T  ... but shapes:
    # K is (N, n_obs), innovation is (M, n_obs)
    # update = K @ innovation.T  -> (N, M), then transpose to (M, N)
    
    update = (K @ innovation.T).T  # (M, N)
    result = result + update
    
    # --- Adaptive Inflation ---
    # Compute per-variable spread vs. innovation-based expected spread
    # Simple adaptive inflation: inflate if ensemble spread is too small relative to innovations
    
    # Compute ensemble spread for observed variables
    ens_spread_obs = np.std(result[:, obs_idx], axis=0, ddof=1)  # (n_obs,)
    
    # Expected spread from observation noise
    expected_spread = np.sqrt(obs_var)
    
    # Inflation factor per observed variable
    # If spread is too small, inflate
    min_spread = 0.5 * expected_spread
    inflation_factors = np.ones(n_obs)
    for j in range(n_obs):
        if ens_spread_obs[j] < min_spread:
            inflation_factors[j] = min_spread / max(ens_spread_obs[j], 1e-10)
            inflation_factors[j] = min(inflation_factors[j], 2.0)  # cap
    
    # Apply inflation to all variables (uniform based on average inflation)
    avg_inflation = np.mean(inflation_factors)
    if avg_inflation > 1.0:
        # Inflate around the mean
        result = ens_mean + avg_inflation * (result - ens_mean)
    
    # Safety: clip to valid range
    np.clip(result, -100.0, 100.0, out=result)
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        result = ensemble.copy()
        result[:, obs_idx] += 0.5 * (observations - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
    
    return result

# EVOLVE-BLOCK-END
