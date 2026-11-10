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
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
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
            inflation = 1.0 + 0.08 * (1.0 - ratio)
            inflation = np.clip(inflation, 1.02, 1.10)
        else:
            inflation = 1.02
        
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    else:
        inflation = 1.02

    # Local EKPF Update with stochastic sampling
    window_half = 3
    window_size = 2 * window_half + 1
    
    # Pre-compute localization weights
    dists = np.arange(-window_half, window_half + 1)
    loc_weights = _gaspari_cohn(dists, c=4.0)
    
    # Convert observed_indices to a set for fast lookup
    obs_set = set(observed_indices.tolist())
    # Map global index to observation value
    obs_dict = {}
    for idx, obs_val in zip(observed_indices, observations):
        obs_dict[idx] = obs_val
    
    # Use a fixed random state for reproducibility within a single call
    rng = np.random.RandomState(42)
    
    # Iterate over each state variable i
    for i in range(N):
        # Check deadline
        if time.time() > deadline:
            break
        
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
        
        # Update local mean (Kalman gain adjusted)
        local_mean_new = local_mean + K_local @ innovation
        
        # Analysis covariance: C_a = (I - K H) C_f
        I_mat = np.eye(window_size)
        KH = K_local @ H_local
        C_a = (I_mat - KH) @ C_local
        
        # Ensure C_a is symmetric and positive semi-definite
        C_a = (C_a + C_a.T) / 2.0
        
        # Add small jitter to ensure positive definiteness
        jitter = 1e-6 * np.eye(window_size)
        C_a_jittered = C_a + jitter
        
        # EKPF: Sample new anomalies from N(0, C_a)
        # Use Cholesky decomposition for sampling
        try:
            L_a = np.linalg.cholesky(C_a_jittered)
            # Generate M samples from N(0, I)
            Z = rng.standard_normal((M, window_size))
            local_anoms_new = Z @ L_a.T
        except np.linalg.LinAlgError:
            # Fallback: use SVD
            try:
                U, s, Vt = np.linalg.svd(C_a_jittered)
                L_a = U @ np.diag(np.sqrt(np.maximum(s, 1e-10)))
                Z = rng.standard_normal((M, window_size))
                local_anoms_new = Z @ L_a.T
            except np.linalg.LinAlgError:
                # Last resort: keep original anomalies scaled
                local_anoms_new = local_anoms * 0.95
        
        # Update ensemble with stochastic anomalies
        local_ens_new = local_mean_new + local_anoms_new
        
        result[:, local_indices] = local_ens_new
    
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
