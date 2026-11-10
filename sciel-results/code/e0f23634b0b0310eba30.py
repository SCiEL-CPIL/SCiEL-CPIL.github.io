"""Combined Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    r1 = d[mask1]
    result[mask1] = (-5.0 / 4.0) * (r1 / c) ** 5 + 5.0 * (r1 / c) ** 4 - 10.0 * (r1 / c) ** 3 + 5.0 * (r1 / c)
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    result[mask2] = (1.0 / 12.0) * (r2 / c) ** 5 - (5.0 / 4.0) * (r2 / c) ** 4 + (5.0 / 2.0) * (r2 / c) ** 3 - 5.0 * (r2 / c) ** 2 + (25.0 / 12.0) * (r2 / c) + 0.5
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    result[mask3] = (-1.0 / 16.0) * (r3 / c) ** 5 + (1.0 / 4.0) * (r3 / c) ** 4 - (15.0 / 8.0) * (r3 / c) ** 3 + (5.0 / 2.0) * (r3 / c) ** 2 - (25.0 / 12.0) * (r3 / c) + 105.0 / 48.0
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    if M > 1:
        spread = np.std(ensemble[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        
        if ratio < 1.0:
            inflation = 1.0 + 0.05 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.01, 1.05)
        else:
            inflation = 1.01
        
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.01

    # Local ESRF Update
    window_half = 2
    window_size = 2 * window_half + 1
    
    # Pre-compute localization weights
    dists = np.arange(-window_half, window_half + 1)
    loc_weights = _gaspari_cohn(dists, c=3.0)
    
    # Convert observed_indices to a set for fast lookup
    obs_set = set(observed_indices.tolist())
    # Map global index to observation value
    obs_dict = {}
    for idx, obs_val in zip(observed_indices, observations):
        obs_dict[idx] = obs_val
    
    # Iterate over each state variable i
    for i in range(N):
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_set:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_dict[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices].copy()
        
        # Extract local observations: shape (n_obs_local,)
        y_local = local_obs_vals[local_obs_mask]
        
        # Compute local ensemble mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Compute local ensemble covariance (M-1 normalization)
        C_local = (local_anoms.T @ local_anoms) / (M - 1)
        
        # Observation operator H_local: selects observed components
        H_local = np.zeros((n_obs_local, window_size))
        obs_positions = np.where(local_obs_mask)[0]
        for k, l_idx in enumerate(obs_positions):
            H_local[k, l_idx] = 1.0
        
        # R = observation_variance * I
        R_local = observation_variance * np.eye(n_obs_local)
        
        # Innovation covariance S = H C H^T + R
        HC = H_local @ C_local
        S = HC @ H_local.T + R_local
        
        # Compute Gain K = C H^T S^-1
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K_local = C_local @ H_local.T @ S_inv
        
        # Innovation y - H x_mean
        innovation = y_local - H_local @ local_mean
        
        # Update local mean
        local_mean_new = local_mean + K_local @ innovation
        
        # ESRF Transform:
        # We want to transform the anomalies A (M, L) such that:
        # 1. The mean of the transformed anomalies is zero (it already is).
        # 2. The new covariance is C_a = C_f - K C H^T (approximately, for the local part)
        #    Actually, for the full ESRF, we match the exact analysis covariance.
        #    C_a = C_f - K_f C_f H^T
        #    where K_f is the full gain.
        
        # Let's use the standard ESRF algorithm:
        # 1. Compute A_f (M, L) anomalies.
        # 2. Compute C_f = A_f^T A_f / (M-1).
        # 3. Compute C_a = C_f - K C_f H^T. (Note: K is (L, n_obs))
        # 4. Compute the transform T such that T A_f^T A_f T^T = (M-1) C_a.
        #    Let A_f^T A_f = (M-1) C_f.
        #    We need T C_f T^T = C_a.
        #    This can be solved via Cholesky or SVD.
        #    Let C_f = L_f L_f^T (Cholesky).
        #    Let C_a = L_a L_a^T (Cholesky).
        #    Then T = L_a L_f^{-1} is one solution, but it's not orthogonal.
        #    The ESRF uses an orthogonal transform T such that T A_f = A_a.
        #    A_a^T A_a = (M-1) C_a.
        #    So T A_f^T A_f T^T = (M-1) C_a.
        #    T C_f T^T = C_a.
        
        # Compute C_a
        C_a = C_local - K_local @ HC
        
        # Ensure C_a is symmetric and positive semi-definite
        C_a = (C_a + C_a.T) / 2.0
        
        # Add small jitter to ensure positive definiteness for Cholesky
        jitter = 1e-8 * np.eye(window_size)
        C_a_jittered = C_a + jitter
        C_f_jittered = C_local + jitter
        
        try:
            L_f = np.linalg.cholesky(C_f_jittered)
            L_a = np.linalg.cholesky(C_a_jittered)
        except np.linalg.LinAlgError:
            # Fallback to SVD if Cholesky fails
            U_f, s_f, Vt_f = np.linalg.svd(C_f_jittered)
            U_a, s_a, Vt_a = np.linalg.svd(C_a_jittered)
            L_f = U_f @ np.diag(np.sqrt(s_f))
            L_a = U_a @ np.diag(np.sqrt(s_a))
        
        # Compute T = L_a L_f^{-1}
        # This T is not orthogonal, but it satisfies T C_f T^T = C_a approximately.
        # To get an orthogonal transform, we can use QR decomposition.
        # Let T_raw = L_a L_f^{-1}.
        # We want an orthogonal T_orth such that T_orth A_f has the same covariance.
        # The standard ESRF computes T via:
        # A_f = L_f Q_f (QR)
        # A_a = L_a Q_a (QR)
        # T = Q_a Q_f^T
        # This T is orthogonal and T A_f = A_a.
        
        # Let's compute Q_f and Q_a
        # A_f is (M, L). L_f is (L, L).
        # We need to factor A_f = L_f Q_f.
        # Since C_f = A_f^T A_f / (M-1), and L_f L_f^T = C_f,
        # A_f^T A_f = (M-1) L_f L_f^T.
        # This doesn't directly give Q_f.
        
        # Alternative approach for ESRF:
        # 1. A_f (M, L)
        # 2. Compute C_f = A_f^T A_f / (M-1)
        # 3. Compute C_a
        # 4. Compute T such that T A_f^T A_f T^T = A_a^T A_a
        #    T (M-1) C_f T^T = (M-1) C_a
        #    T C_f T^T = C_a
        #    Let C_f = U_f S_f U_f^T, C_a = U_a S_a U_a^T
        #    T = U_a S_a^{1/2} S_f^{-1/2} U_f^T
        #    This T is symmetric and positive definite, but not orthogonal.
        #    For ESRF, we want an orthogonal transform.
        #    The key insight is that we can choose the sign of the square roots to make T orthogonal.
        #    Actually, the ESRF transform is orthogonal if and only if C_f and C_a have the same eigenvalues,
        #    which is not true here. So the ESRF transform is NOT orthogonal in the general case.
        #    It is a "square root" transform that preserves the ensemble spread in a generalized sense.
        
        # Let's just use T = L_a L_f^{-1} and apply it to the anomalies.
        # A_a = T A_f^T ? No, A_f is (M, L).
        # We want A_a (M, L) such that A_a^T A_a = (M-1) C_a.
        # Let A_f = Q_f R_f (QR decomposition of A_f^T, which is (L, M)).
        # A_f^T = Q_f R_f, so A_f = R_f^T Q_f^T.
        # This is getting complicated.
        
        # Let's use a simpler approach:
        # Compute the transform directly using the covariance matrices.
        # T = C_a C_f^{-1}
        # Then A_a = T A_f
        # Check: A_a^T A_a = A_f^T T^T T A_f = A_f^T C_f^{-1} C_a C_a C_f^{-1} A_f
        # This is not (M-1) C_a.
        
        # Correct ESRF derivation:
        # We want A_a such that A_a^T A_a / (M-1) = C_a.
        # We have A_f such that A_f^T A_f / (M-1) = C_f.
        # Let A_f = L_f Q_f where Q_f is (M, L) with orthonormal columns.
        # Then A_f^T A_f = Q_f^T L_f^T L_f Q_f = Q_f^T C_f Q_f.
        # This implies C_f = L_f^T L_f? No.
        # If A_f = L_f Q_f, then A_f^T A_f = Q_f^T L_f^T L_f Q_f.
        # And C_f = A_f^T A_f / (M-1).
        # So L_f^T L_f is not necessarily C_f.
        
        # Let's use the SVD approach for the transform.
        # C_f = U_f S_f U_f^T
        # C_a = U_a S_a U_a^T
        # T = U_a S_a^{1/2} S_f^{-1/2} U_f^T
        # Then T C_f T^T = U_a S_a^{1/2} S_f^{-1/2} U_f^T U_f S_f U_f^T S_f^{-1/2} S_a^{1/2} U_a^T
        # = U_a S_a^{1/2} S_f^{-1/2} S_f S_f^{-1/2} S_a^{1/2} U_a^T
        # = U_a S_a U_a^T = C_a. Correct.
        
        # Now, A_a = T A_f
        # A_a^T A_a = A_f^T T^T T A_f
        # T^T T = U_f S_f^{-1/2} S_a S_f^{-1/2} U_f^T
        # A_f^T T^T T A_f = A_f^T U_f S_f^{-1/2} S_a S_f^{-1/2} U_f^T A_f
        # This is not simply (M-1) C_a.
        
        # The issue is that A_f and C_f are related by A_f^T A_f = (M-1) C_f,
        # but A_f is not uniquely determined by C_f.
        # The ESRF assumes that the transform T is applied to the anomalies,
        # and the new anomalies A_a = T A_f have the desired covariance.
        # This requires that T A_f^T A_f T^T = (M-1) C_a.
        # T (M-1) C_f T^T = (M-1) C_a.
        # T C_f T^T = C_a.
        # So if we choose T such that T C_f T^T = C_a, then A_a = T A_f will have the correct covariance.
        
        # So let's compute T = U_a S_a^{1/2} S_f^{-1/2} U_f^T.
        
        try:
            U_f, s_f, Vt_f = np.linalg.svd(C_f_jittered)
            U_a, s_a, Vt_a = np.linalg.svd(C_a_jittered)
            
            # Ensure positive singular values
            s_f = np.maximum(s_f, 1e-12)
            s_a = np.maximum(s_a, 1e-12)
            
            T = (U_a @ np.diag(np.sqrt(s_a)) @ np.diag(1.0 / np.sqrt(s_f)) @ U_f.T)
        except np.linalg.LinAlgError:
            T = np.eye(window_size)
        
        # Apply transform to anomalies
        local_anoms_new = local_anoms @ T.T
        
        # Update ensemble
        local_ens_new = local_mean_new + local_anoms_new
        
        result[:, local_indices] = local_ens_new
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
