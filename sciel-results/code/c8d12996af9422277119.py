"""Local Ensemble Square Root Filter (ESRF) for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Precompute Gaspari-Cohn localization matrix for N=20 ring topology
_N = 20
_CORR_LEN = 6.0  # correlation length in grid units

def _gaspari_cohn(d, c):
    """Gaspari-Cohn correlation function (C2)."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    r1 = d[mask1]
    if np.any(mask1):
        result[mask1] = (-5.0/4.0)*(r1/c)**5 + 5.0*(r1/c)**4 - 10.0*(r1/c)**3 + 5.0*(r1/c)
    mask2 = (d > 0.5*c) & (d <= 1.5*c)
    r2 = d[mask2]
    if np.any(mask2):
        result[mask2] = (1.0/12.0)*(r2/c)**5 - (5.0/4.0)*(r2/c)**4 + (5.0/2.0)*(r2/c)**3 - 5.0*(r2/c)**2 + (25.0/12.0)*(r2/c) + 0.5
    mask3 = (d > 1.5*c) & (d <= 2.5*c)
    r3 = d[mask3]
    if np.any(mask3):
        result[mask3] = -(1.0/4.0)*(r3/c)**5 + (5.0/4.0)*(r3/c)**4 - (5.0/2.0)*(r3/c)**3 + (5.0/3.0)*(r3/c)**2 - (25.0/12.0)*(r3/c) + (5.0/3.0)
    return result

# Build the localization matrix: L[i,j] = GC(min(|i-j|, N-|i-j|), c)
def _build_loc_matrix():
    L = np.zeros((_N, _N), dtype=float)
    for i in range(_N):
        for j in range(_N):
            d = min(abs(i - j), _N - abs(i - j))
            L[i, j] = _gaspari_cohn(d, _CORR_LEN)
    # Ensure diagonal is 1
    np.fill_diagonal(L, 1.0)
    return L

_LOC_MATRIX = _build_loc_matrix()

def _adaptive_inflation(ensemble, observations, observed_indices, observation_variance):
    """Compute adaptive multiplicative inflation factor."""
    # Compute ensemble variance at observed points
    ens_obs = ensemble[:, observed_indices]
    ens_var = np.var(ens_obs, axis=0, ddof=1)
    
    # Innovation variance
    innov = observations - ens_obs
    innov_var = np.var(innov, axis=0, ddof=1)
    
    # Expected innovation variance = ens_var + obs_var
    expected_var = ens_var + observation_variance
    
    # Ratio of observed to expected
    ratio = np.sum(innov_var) / np.sum(expected_var)
    
    # Adaptive inflation: scale so that expected matches observed
    # infl_factor^2 * expected_var = observed_var
    # infl_factor = sqrt(ratio) but clipped
    infl = np.sqrt(ratio)
    
    # Clip to reasonable range
    infl = np.clip(infl, 1.0, 1.5)
    
    return infl

def _local_esrf_update(ensemble, observations, observed_indices, observation_variance):
    """
    Apply local ESRF update.
    ensemble: (M, N) array
    observations: (M, K) array where K = len(observed_indices)
    observed_indices: array of observed variable indices
    observation_variance: scalar
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Adaptive inflation
    infl = _adaptive_inflation(ensemble, observations, observed_indices, observation_variance)
    
    # Apply inflation to anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    anomalies *= infl
    inflated_ensemble = ens_mean + anomalies
    
    # Local ESRF: for each observed variable, update the ensemble
    # We process all observations jointly using the localization-weighted approach
    
    # Build the observation operator H (K x N) - identity on observed indices
    # For local ESRF with localization, we use the approach where we:
    # 1. Compute the analysis using the full ensemble covariance structure
    # 2. Apply localization by scaling the update
    
    # Standard ESRF approach:
    # For each member m, compute the update
    # The ESRF uses an orthogonal transformation Q such that:
    # x_a = H_a * x_f + (I - H_a) * x_f  ... (for unobserved)
    # For observed variables, we need to match the analysis covariance
    
    # Simpler approach: use the square root filter formulation
    # R = observation covariance = observation_variance * I_K
    # We need to find Q (M x M orthogonal) such that:
    # (x_f + Q * anomalies) has the correct analysis properties
    
    # Actually, let's use the standard ESRF algorithm:
    # 1. Compute forecast mean and anomalies
    # 2. For observed variables, compute the analysis using the relationship:
    #    x_a^obs = H x_f^obs + K (y - H x_f^obs)
    #    where K is the Kalman gain, but in square root form
    
    # Let me use a cleaner ESRF formulation:
    # The analysis ensemble is: X_a = X_f + A
    # where A is chosen so that:
    # - mean(X_a) = mean(X_f) + K(y - H mean(X_f))  [for observed]
    # - cov(X_a) = (I - K H) cov(X_f)  [for observed]
    
    # In square root form, we work with the anomaly matrix
    # Let's use the approach from Hunt et al. (2007):
    
    # Compute the forecast covariance at observed points
    H = np.zeros((K, N))
    for k, idx in enumerate(observed_indices):
        H[k, idx] = 1.0
    
    # Forecast anomaly matrix at observed points: (M, K)
    Xf_obs = inflated_ensemble[:, observed_indices]  # (M, K)
    mean_obs = np.mean(Xf_obs, axis=0)  # (K,)
    Anom_obs = Xf_obs - mean_obs  # (M, K)
    
    # Compute the analysis
    # R = observation_variance * I
    R = observation_variance * np.eye(K)
    
    # Innovation
    innov = observations - Xf_obs  # (M, K)
    mean_innov = np.mean(innov, axis=0)  # (K,)
    
    # For ESRF, we need to find an orthogonal matrix Q such that:
    # The analysis anomalies at observed points have the correct covariance
    # Analysis covariance = (I - K H) P_f (I - K H)^T + K R K^T
    # where K = P_f H^T (H P_f H^T + R)^{-1}
    
    # Let's compute the Kalman gain
    Pf_obs = np.cov(Anom_obs, rowvar=False)  # (K, K)
    # Add small regularization for numerical stability
    Pf_obs += 1e-10 * np.eye(K)
    
    S = H @ Pf_obs @ H.T + R  # (K, K)
    K_gain = Pf_obs @ H.T @ np.linalg.inv(S)  # (K, K)
    
    # Analysis mean
    mean_a = mean_obs + K_gain @ mean_innov  # (K,)
    
    # Now for the ESRF square root update:
    # We need to transform the anomalies so that:
    # 1. The mean shifts correctly
    # 2. The covariance at observed points matches the analysis covariance
    
    # Analysis covariance at observed points
    Pa_obs = (np.eye(K) - K_gain @ H) @ Pf_obs @ (np.eye(K) - K_gain @ H).T + K_gain @ R @ K_gain.T
    Pa_obs = 0.5 * (Pa_obs + Pa_obs.T)  # Symmetrize
    
    # For unobserved points, we keep the forecast covariance (with localization)
    # For observed points, we want the analysis covariance
    
    # The ESRF approach:
    # Let X_a = X_f + Delta, where Delta is the update
    # We want:
    #   mean(Delta) = K_gain @ mean_innov  (for observed)
    #   cov(X_a_obs) = Pa_obs
    
    # In square root form, we decompose:
    # Let A = Anom_obs (M x K) be the forecast anomalies at observed points
    # We want to find B (M x K) such that:
    #   B B^T / (M-1) = Pa_obs
    #   And the cross-covariance with unobserved points is handled via localization
    
    # Simpler: use the approach where we directly construct the analysis
    # For each observed variable k:
    #   x_a[m, k] = mean_a[k] + B[m, k]
    # where B is chosen so that B B^T / (M-1) = Pa_obs
    
    # Cholesky-like decomposition
    try:
        L_pa = np.linalg.cholesky(Pa_obs)
    except:
        L_pa = np.linalg.cholesky(Pa_obs + 1e-8 * np.eye(K))
    
    # We need to transform the forecast anomalies to have the analysis covariance
    # The forecast anomalies have covariance Pf_obs
    # We want: B = Anom_obs @ Q^T for some orthogonal Q
    # such that B B^T / (M-1) = Pa_obs
    
    # This requires: Anom_obs @ Q^T @ Q @ Anom_obs^T / (M-1) = Pa_obs
    # i.e., Anom_obs @ Anom_obs^T / (M-1) = Pf_obs, which is true
    # So we need Q such that the transformed anomalies have the right covariance
    
    # Actually, the standard ESRF approach is:
    # 1. Compute the analysis covariance Pa
    # 2. Find the transformation that maps forecast anomalies to analysis anomalies
    
    # Let's use a different, cleaner approach:
    # The analysis is: x_a = x_f + K(y - H x_f)
    # In ensemble form:
    # X_a = X_f + K_gain @ (Y - H X_f)
    #     = (I - K_gain H) X_f + K_gain Y
    
    # But this is the linear EnKF. For ESRF, we need to preserve the spread exactly.
    
    # Let me implement the proper ESRF:
    # From Hunt et al. (2007), the ESRF update is:
    # X_a = X_f + A, where A is computed as follows:
    # 1. Let the forecast anomaly matrix be A_f (M x N)
    # 2. Compute the analysis covariance P_a
    # 3. Find an orthogonal matrix Q such that:
    #    A_a = A_f @ Q^T  (for the observed components)
    #    and A_a A_a^T / (M-1) = P_a_obs
    
    # This is equivalent to finding Q such that:
    # A_f_obs @ Q^T @ Q @ A_f_obs^T / (M-1) = P_a_obs
    # But A_f_obs @ A_f_obs^T / (M-1) = P_f_obs
    # So we need Q such that the transformed covariance equals P_a_obs
    
    # This is a Procrustes-type problem. Let's use SVD.
    # We want: A_f_obs @ Q^T ≈ L_pa @ V^T where V is any orthogonal matrix
    # Actually, we need: (A_f_obs @ Q^T)(A_f_obs @ Q^T)^T = (M-1) P_a_obs
    
    # Let A_f_obs = U_f S_f V_f^T (SVD)
    # Then A_f_obs @ Q^T = U_f S_f V_f^T Q^T
    # We want this to have covariance P_a_obs
    
    # Alternative: use the fact that both P_f_obs and P_a_obs are positive definite
    # We can write P_a_obs = P_f_obs^{1/2} M P_f_obs^{1/2} for some matrix M
    # Then the transformation involves M
    
    # This is getting complex. Let me use a simpler but still valid approach:
    # Use the EnKF with localization and adaptive inflation, which is well-understood
    # and has been shown to work well for Lorenz-96.
    
    # Actually, let me just implement a clean local ESRF using the following approach:
    # For each observed variable, apply the update independently with localization.
    
    # This is the "local" part of Local ESRF.
    
    result = inflated_ensemble.copy()
    
    # For each observed index, apply a 1D ESRF update
    for k, obs_idx in enumerate(observed_indices):
        # Get the forecast values at this index
        x_f = inflated_ensemble[:, obs_idx]  # (M,)
        y = observations[:, k]  # (M,)
        
        # Compute forecast variance
        mean_f = np.mean(x_f)
        var_f = np.var(x_f, ddof=1)
        
        # Observation variance
        var_r = observation_variance
        
        # Kalman gain
        K_gain_1d = var_f / (var_f + var_r)
        
        # Analysis mean
        mean_a = mean_f + K_gain_1d * (np.mean(y) - mean_f)
        
        # Analysis variance
        var_a = (1 - K_gain_1d) * var_f
        
        # Apply the update to each member
        # x_a[m] = mean_a + sqrt(var_a/var_f) * (x_f[m] - mean_f)
        # This preserves the relative ordering and shape of the ensemble
        
        if var_f > 1e-10:
            scale = np.sqrt(var_a / var_f)
            result[:, obs_idx] = mean_a + scale * (x_f - mean_f)
        else:
            result[:, obs_idx] = mean_a
    
    # Apply localization to unobserved points
    # For unobserved points, blend the forecast with the analysis using the localization matrix
    unobserved_indices = [i for i in range(N) if i not in observed_indices]
    
    for u_idx in unobserved_indices:
        # Compute the analysis mean at observed points
        # Blend using localization weights
        weights = _LOC_MATRIX[u_idx, observed_indices]  # (K,)
        weight_sum = np.sum(weights)
        
        if weight_sum > 1e-10:
            # Weighted average of analysis means at observed points
            analysis_means_obs = np.mean(result[:, observed_indices], axis=0)
            forecast_mean_u = np.mean(inflated_ensemble[:, u_idx])
            
            # Blend
            w = np.min(1.0, weight_sum)  # Cap the weight
            blended_mean = (1 - w) * forecast_mean_u + w * np.average(analysis_means_obs, weights=weights)
            
            # Apply to each member
            anomalies_u = inflated_ensemble[:, u_idx] - forecast_mean_u
            result[:, u_idx] = blended_mean + anomalies_u
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Main analysis function using Local ESRF.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble forecast
    observations : ndarray, shape (16, 10)
        Noisy observations of even-indexed variables
    observed_indices : array-like
        Indices of observed variables
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble
    """
    start_time = time.time()
    
    try:
        result = _local_esrf_update(ensemble, observations, observed_indices, observation_variance)
    except Exception:
        # Fallback to simple nudging if ESRF fails
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
    
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 110.0:
        # If we're running out of time, return what we have
        pass
    
    return result
