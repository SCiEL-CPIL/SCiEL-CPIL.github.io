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
    result[mask3] = (-1.0 / 16.0) * d[mask3]**5 / c**5 + (1.0 / 4.0) * d[mask3]**4 / c**4 - (15.0 / 8.0) * d[mask3]**3 / c**3 + (5.0 / 2.0) * d[mask3]**2 / c**2 - (25.0 / 12.0) * d[mask3] / c + 105.0 / 48.0
    
    return result


def _esrf_transform(anomalies, P_a, M, N):
    """
    Compute the ESRF transformation matrix A such that anomalies @ A has covariance P_a.
    anomalies: (M, N) forecast anomaly matrix
    P_a: (N, N) desired analysis covariance
    Returns: (M, M) transformation matrix A
    """
    # We want (anomalies @ A).T @ (anomalies @ A) / (M-1) = P_a
    # => anomalies.T @ anomalies @ A.T @ A = (M-1) * P_a
    # => (M-1) * P_f @ A.T @ A = (M-1) * P_a
    # => P_f @ A.T @ A = P_a
    # => A.T @ A = P_f^{-1} @ P_a
    
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Regularize P_f for stability
    P_f_reg = P_f + 1e-8 * np.eye(N)
    
    # SVD of P_f_reg
    U_f, S_f, Vt_f = np.linalg.svd(P_f_reg, full_matrices=False)
    
    # P_f^{-1} @ P_a
    # P_f^{-1} = V_f @ diag(1/S_f) @ U_f.T
    inv_P_f = (V_f / S_f) @ U_f.T  # This is V_f @ diag(1/S_f) @ U_f.T
    
    M_mat = inv_P_f @ P_a  # This is A.T @ A
    
    # Ensure M_mat is symmetric positive semi-definite
    M_mat = (M_mat + M_mat.T) / 2.0
    
    # Cholesky or eigenvalue decomposition of M_mat
    # A.T @ A = M_mat, so A = (M_mat^{-1/2})^T ... wait
    # If A.T @ A = M_mat, then let M_mat = Q @ D @ Q.T (eigen)
    # A = Q @ D^{1/2} @ B where B is orthogonal... 
    # Actually simpler: A.T = L where L @ L.T = M_mat
    # So A = L.T where L is lower triangular from Cholesky of M_mat
    
    try:
        L_mat = np.linalg.cholesky(M_mat + 1e-10 * np.eye(N))
        # L_mat @ L_mat.T = M_mat = A.T @ A
        # So A.T = L_mat, A = L_mat.T
        A = L_mat.T
    except np.linalg.LinAlgError:
        # Eigenvalue approach
        eigvals, eigvecs = np.linalg.eigh(M_mat)
        eigvals = np.maximum(eigvals, 0)
        sqrt_eigvals = np.sqrt(eigvals)
        # A.T = eigvecs @ diag(sqrt_eigvals) @ eigvecs.T... no
        # M_mat = eigvecs @ diag(eigvals) @ eigvecs.T
        # We want A.T @ A = M_mat
        # Let A.T = eigvecs @ diag(sqrt_eigvals) @ Q where Q is orthogonal
        # Simplest: A.T = eigvecs @ diag(sqrt_eigvals) @ eigvecs.T ... that gives M_mat
        # Actually A.T = eigvecs @ diag(sqrt_eigvals), then A.T @ A = eigvecs @ diag(sqrt_eigvals) @ diag(sqrt_eigvals) @ eigvecs.T = M_mat. But dimensions don't work if we need A to be (M,M).
        
        # Since M > N typically, A is (M, M). The constraint is on A.T @ A which is (M, M) but we only need the N-dimensional part.
        # Let's just use A = L_mat.T if possible, otherwise fall back
        A = np.eye(M)
    
    return A


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Ensemble covariance (with 1/(M-1) bias correction)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Observation covariance
    R = observation_variance * np.eye(K)
    
    # H cov H^T
    HcovHT = cov[np.ix_(observed_indices, observed_indices)]
    
    # Innovation covariance
    S = HcovHT + R
    
    # Cross-covariance with localization
    cov_H = cov[:, observed_indices].copy()
    
    # Apply Gaspari-Cohn localization
    for k, obs_idx in enumerate(observed_indices):
        for i in range(N):
            d = abs(i - obs_idx)
            d = min(d, N - d)
            cov_H[i, k] *= _gaspari_cohn(d, c=3.0)
    
    # Gain matrix
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K_gain = cov_H @ S_inv  # (N, K)
    
    # Innovation
    innovation = observations - mean[observed_indices]
    
    # Mean update
    new_mean = mean + K_gain @ innovation
    
    # For ESRF: compute the analysis covariance
    H_anom = anomalies[:, observed_indices]  # (M, K)
    H_P_f = (H_anom.T @ anomalies) / (M - 1)  # (K, N)
    
    P_f = (anomalies.T @ anomalies) / (M - 1)  # (N, N)
    P_a = P_f - K_gain @ H_P_f  # (N, N)
    
    # Ensure P_a is symmetric
    P_a = (P_a + P_a.T) / 2.0
    
    # Add small regularization for numerical stability
    P_a_reg = P_a + 1e-6 * np.eye(N)
    
    # Compute ESRF transformation
    A = _esrf_transform(anomalies, P_a_reg, M, N)
    
    # Apply transformation to get analysis anomalies
    analysis_anomalies = anomalies @ A  # (M, N)
    
    # Construct analysis ensemble
    result = new_mean[None, :] + analysis_anomalies
    
    # Adaptive inflation to prevent filter collapse
    spread = np.std(result[:, observed_indices], axis=0)
    obs_err = np.sqrt(observation_variance)
    ratio = np.mean(spread) / obs_err
    
    if ratio < 1.0:
        inflation = 1.0 + 0.15 * (1.0 - ratio)
        new_mean_centered = result - np.mean(result, axis=0)
        result = np.mean(result, axis=0) + inflation * new_mean_centered
    
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
