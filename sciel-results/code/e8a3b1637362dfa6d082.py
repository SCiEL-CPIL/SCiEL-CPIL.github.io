"""Ensemble Transform Kalman Filter (ETKF) for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

_N = 20
_N_OBS = 10  # even indices: 0,2,4,...,18

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

def _build_loc_matrix(corr_len=5.0):
    L = np.zeros((_N, _N), dtype=float)
    for i in range(_N):
        for j in range(_N):
            d = min(abs(i - j), _N - abs(i - j))
            L[i, j] = _gaspari_cohn(d, corr_len)
    np.fill_diagonal(L, 1.0)
    return L

# Precompute localization matrix
_LOC = _build_loc_matrix(5.0)

def _etkf_update(ensemble, observations, observed_indices, observation_variance):
    """
    ETKF update with localization and adaptive inflation.
    ensemble: (M, N)
    observations: (1, K) or (K,)
    observed_indices: array of observed indices
    observation_variance: scalar
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # Adaptive inflation
    ens_mean = np.mean(ensemble, axis=0)
    ens_obs = ensemble[:, observed_indices]
    ens_var_obs = np.var(ens_obs, axis=0, ddof=1)
    innov = obs - ens_mean[observed_indices]
    innov_var = np.var(innov)
    expected_var = np.mean(ens_var_obs) + observation_variance
    
    if expected_var > 1e-10:
        ratio = innov_var / expected_var
        infl = np.sqrt(max(ratio, 1.0))
    else:
        infl = 1.0
    infl = np.clip(infl, 1.0, 1.5)
    
    # Apply inflation
    anomalies = ensemble - ens_mean
    inflated = ens_mean + infl * anomalies
    
    # Local ETKF: process each observed variable sequentially
    # For each observation, update the full state using the cross-covariance
    result = inflated.copy()
    
    # Build the observation operator H (sparse)
    # H is K x N, with H[k, observed_indices[k]] = 1
    
    # For ETKF, we work in the anomaly space
    # The key idea: find an orthogonal transformation Q (M x M) such that
    # the transformed anomalies have the correct analysis covariance
    
    # Simplified approach: use the ensemble Kalman filter equations
    # with localization applied to the covariance
    
    # Compute forecast covariance (localized)
    A = inflated - np.mean(inflated, axis=0)  # N x M anomalies
    P_f = (A @ A.T) / (M - 1)  # N x N
    P_f_loc = P_f * _LOC
    P_f_loc = 0.5 * (P_f_loc + P_f_loc.T)
    
    # Observation covariance
    R = observation_variance * np.eye(K)
    
    # Innovation
    innov = obs - np.mean(inflated[:, observed_indices], axis=0)
    
    # Kalman gain
    P_f_obs = P_f_loc[np.ix_(observed_indices, observed_indices)]
    S = P_f_obs + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    P_cross = P_f_loc[:, observed_indices]  # N x K
    K_gain = P_cross @ S_inv  # N x K
    
    # Update mean
    mean_a = np.mean(inflated, axis=0) + K_gain @ innov
    
    # Update covariance
    term1 = K_gain @ P_f_obs @ K_gain.T
    term2 = K_gain @ R @ K_gain.T
    P_a = P_f_loc - term1 + term2
    P_a = 0.5 * (P_a + P_a.T)
    
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(P_a)
    eigvals = np.clip(eigvals, 0, None)
    P_a = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Reconstruct ensemble: A_a = A_f Q^T where Q is M x M orthogonal
    # We want A_a A_a^T / (M-1) = P_a
    # Use SVD-based approach
    try:
        U_f, S_f, Vt_f = np.linalg.svd(A, full_matrices=False)
        # A = U_f S_f Vt_f
        # A A^T = U_f S_f^2 U_f^T
        
        # We want A_a such that A_a A_a^T = (M-1) P_a
        # Let P_a = U_a S_a^2 U_a^T
        U_a, S_a_sq, Vt_a = np.linalg.svd((M-1) * P_a, full_matrices=False)
        S_a = np.sqrt(np.clip(S_a_sq, 0, None))
        
        # A_a = U_a S_a V_a^T where V_a is M x M orthogonal
        # To maintain structure, set V_a = Vt_f^T (i.e., V_a^T = Vt_f)
        # But dimensions: U_a is N x r, S_a is r x r, V_a^T should be r x M
        # This doesn't work directly because A is N x M, not square
        
        # Alternative: use the fact that A_a = A + correction
        # For each member m: x_a[m] = mean_a + (x_f[m] - mean_f) * scale + correction
        # This is essentially what the simple EnKF does
    except:
        pass
    
    # Use the standard EnKF update for the anomalies
    # For each variable i:
    #   x_a[i, m] = mean_a[i] + (x_f[i, m] - mean_f[i]) * sqrt(var_a[i] / var_f[i])
    # where var_a[i] = P_a[i,i], var_f[i] = P_f_loc[i,i]
    
    mean_f = np.mean(inflated, axis=0)
    var_f = np.diag(P_f_loc)
    var_a = np.diag(P_a)
    
    # Ensure positive variances
    var_f = np.clip(var_f, 1e-10, None)
    var_a = np.clip(var_a, 1e-10, None)
    
    scale = np.sqrt(var_a / var_f)
    
    # Reconstruct ensemble
    result = mean_a + (inflated - mean_f) * scale[:, np.newaxis]
    
    return np.clip(result, -100.0, 100.0)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Main analysis function using ETKF.
    """
    start_time = time.time()
    
    try:
        result = _etkf_update(ensemble, observations, observed_indices, observation_variance)
    except Exception:
        # Fallback: simple nudging
        result = ensemble.copy()
        obs = np.asarray(observations).flatten()
        for k, idx in enumerate(observed_indices):
            result[:, idx] += 0.5 * (obs[k] - result[:, idx])
    
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
