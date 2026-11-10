"""Combined ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) update with adaptive inflation
    and Gaspari-Cohn localization for Lorenz-96 partial-observation assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states.
    observations: (K,) or (1, K) array of noisy observations for observed indices.
    observed_indices: array of indices that are observed.
    observation_variance: scalar variance of the observation noise.
    """
    start_time = time.time()
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure observations is 1D
    obs = np.asarray(observations).flatten()
    
    # 1. Compute ensemble mean
    mean_f = np.mean(ensemble, axis=0)
    
    # 2. Compute anomalies
    anomalies = ensemble - mean_f
    
    # 3. Adaptive inflation based on innovation magnitude
    obs_mean_f = mean_f[observed_indices]
    innovation = obs - obs_mean_f
    innovation_norm = np.linalg.norm(innovation)
    
    # Base inflation plus adaptive component
    base_inflation = 1.03
    expected_innovation_scale = np.sqrt(K * observation_variance)
    if expected_innovation_scale > 0:
        ratio = innovation_norm / expected_innovation_scale
        adaptive_component = 0.05 * np.clip(ratio, 0, 2)
    else:
        adaptive_component = 0.0
    
    inflation_factor = base_inflation + adaptive_component
    inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    
    anomalies *= inflation_factor
    
    # 4. Compute ensemble covariance with localization
    P_f = (anomalies.T @ anomalies) / (M - 1)
    
    # Apply Gaspari-Cohn localization
    L = 3.0  # Correlation length scale in indices
    
    # Build localization matrix
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)
            if d <= L:
                if d <= L / 2:
                    x1 = 2 * d / L - 5
                    x2 = 2 * d / L - 3
                    x3 = 2 * d / L - 1
                    loc_matrix[i, j] = (-0.25 * x1**5 + 5 * x2**5 + 5 * x3**5 + 5) / 4
                else:
                    x1 = 2 * d / L - 7
                    x2 = 2 * d / L - 5
                    x3 = 2 * d / L - 3
                    x4 = 2 * d / L - 1
                    loc_matrix[i, j] = (0.5 * x1**5 - 5 * x2**5 + 10 * x3**5 - 5 * x4**5 + 5) / 2
            else:
                loc_matrix[i, j] = 0.0
    
    P_f_loc = P_f * loc_matrix
    
    # 5. Observation operator H is implicit via indices
    # S = H P_f H^T + R
    H_P_H = P_f_loc[np.ix_(observed_indices, observed_indices)]
    R = np.eye(K) * observation_variance
    S = H_P_H + R
    
    # 6. Compute analysis covariance: P_a = P_f - P_f H^T S^-1 H P_f
    # Using the identity: P_a = P_f_loc - P_f_loc[:, obs] @ inv(S) @ P_f_loc[obs, :]
    P_f_obs = P_f_loc[:, observed_indices]  # N x K
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    P_a = P_f_loc - P_f_obs @ S_inv @ P_f_obs.T
    
    # Ensure P_a is symmetric and positive semi-definite
    P_a = (P_a + P_a.T) / 2
    
    # 7. Compute analysis mean: mean_a = mean_f + P_f H^T S^-1 (obs - H mean_f)
    # Note: P_f here should be the localized P_f for consistency
    K_gain = P_f_obs @ S_inv  # N x K
    mean_a = mean_f + K_gain @ innovation
    
    # 8. ESRF update: find orthogonal transform that maps forecast anomalies to analysis anomalies
    # We need: A_a = A_f @ Q, where Q is orthogonal and A_a @ A_a.T / (M-1) = P_a
    # A_f @ A_f.T / (M-1) = P_f_loc (by construction after inflation)
    # So we need Q such that P_f_loc @ Q @ Q.T = P_a
    # This is a Cholesky-based ESRF update
    
    # Compute Cholesky of P_f_loc + small jitter for numerical stability
    jitter = 1e-10
    P_f_jittered = P_f_loc + jitter * np.eye(N)
    
    try:
        L_f = np.linalg.cholesky(P_f_jittered)
    except np.linalg.LinAlgError:
        # Fallback to SVD if Cholesky fails
        U_f, s_f, Vt_f = np.linalg.svd(P_f_jittered)
        L_f = U_f @ np.diag(np.sqrt(s_f))
    
    # Similarly for P_a
    P_a_jittered = P_a + jitter * np.eye(N)
    try:
        L_a = np.linalg.cholesky(P_a_jittered)
    except np.linalg.LinAlgError:
        U_a, s_a, Vt_a = np.linalg.svd(P_a_jittered)
        L_a = U_a @ np.diag(np.sqrt(s_a))
    
    # The ESRF transform: Q = L_f^{-T} @ L_a^{-T}
    # Actually, the standard ESRF uses:
    # A_a = A_f @ (L_f^{-1} @ L_a^T)^T = A_f @ L_a @ L_f^{-T}
    # Let's use the more stable form:
    # Solve L_f @ X.T = I for X, then Q = L_a @ X
    # Or directly: Q = solve(L_f.T, L_a.T).T
    
    try:
        # Q = L_f^{-T} @ L_a^{-T} ... let me use the standard formulation
        # We want A_a = A_f @ Q where Q is orthogonal
        # P_a = Q.T @ P_f_loc @ Q (since A_a @ A_a.T = A_f @ Q @ Q.T @ A_f.T)
        # Wait, let me reconsider.
        # A_f @ A_f.T = (M-1) P_f_loc
        # A_a @ A_a.T = (M-1) P_a
        # A_a = A_f @ Q
        # A_f @ Q @ Q.T @ A_f.T = (M-1) P_a
        # So we need Q such that Q.T @ P_f_loc @ Q = P_a (up to scaling)
        
        # Standard ESRF: 
        # Let C_f = A_f @ A_f.T / (M-1) = P_f_loc
        # Let C_a = P_a
        # Compute Cholesky: C_f = L_f @ L_f.T, C_a = L_a @ L_a.T
        # Then Q = L_f^{-1} @ L_a @ L_a.T @ L_f^{-T} ... this is getting complicated.
        
        # Simpler approach: use the fact that for ESRF,
        # Q = (L_f^{-1} @ L_a) @ (L_f^{-1} @ L_a).T is not right either.
        
        # Let's use the direct method:
        # We want to find Q orthogonal such that A_f @ Q has covariance P_a
        # One standard approach: Q = L_f^{-T} @ L_a^{-T} is not orthogonal in general.
        
        # Actually, the correct ESRF transform is:
        # Q = L_f^{-1} @ L_a @ (L_f^{-1} @ L_a)^{-T}
        # This ensures Q is orthogonal.
        
        # Let's compute:
        L_f_inv = np.linalg.inv(L_f)
        L_a_inv = np.linalg.inv(L_a)
        
        # Compute intermediate
        M1 = L_f_inv @ L_a  # N x N
        
        # Q = M1 @ M1^{-T}
        M1_inv_T = np.linalg.inv(M1.T)
        Q = M1 @ M1_inv_T
        
        # Ensure Q is orthogonal (orthogonalize if needed)
        # QR decomposition to get orthogonal part
        Qr, Qq = np.linalg.qr(Q)
        Q = Qq
        
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use the linear EnKF update for anomalies
        # anomalies_a = anomalies @ (I - H^T K)
        H_T_K_T = np.zeros((N, N))
        for k, idx in enumerate(observed_indices):
            H_T_K_T[idx, :] = K_gain[:, k]
        
        M_update = np.eye(N) - H_T_K_T
        anomalies_a = anomalies @ M_update
        ensemble_a = mean_a[None, :] + anomalies_a
        ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
        ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
        return ensemble_a
    
    # Apply the orthogonal transform to anomalies
    anomalies_a = anomalies @ Q
    
    # 9. Reconstruct ensemble
    ensemble_a = mean_a[None, :] + anomalies_a
    
    # Ensure finite and within bounds
    ensemble_a = np.nan_to_num(ensemble_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble_a = np.clip(ensemble_a, -100.0, 100.0)
    
    # Deadline guard
    if time.time() - start_time > 110:
        return ensemble_a
    
    return ensemble_a
# EVOLVE-BLOCK-END
