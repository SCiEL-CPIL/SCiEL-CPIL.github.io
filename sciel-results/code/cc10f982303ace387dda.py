"""Lorenz-96 ensemble assimilation with Local ESRF (Ensemble Square Root Filter)."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 3  # window of 7 variables (i-3 to i+3)

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    dm = d[m1]
    if len(dm) > 0:
        result[m1] = (-5.0/4.0)*(dm/c)**5 + 5.0*(dm/c)**4 - 10.0*(dm/c)**3 + 5.0*dm
    
    dm = d[m2]
    if len(dm) > 0:
        result[m2] = (1.0/12.0)*(dm/c)**5 - (5.0/4.0)*(dm/c)**4 + (2.5)*(dm/c)**3 - 5.0*(dm/c)**2 + (25.0/12.0)*(dm/c) + 0.5
    
    dm = d[m3]
    if len(dm) > 0:
        result[m3] = -(1.0/16.0)*(dm/c)**5 + (0.25)*(dm/c)**4 - (15.0/8.0)*(dm/c)**3 + (2.5)*(dm/c)**2 - (25.0/12.0)*(dm/c) + 105.0/48.0
    
    return result

def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ESRF update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Build observed index set for fast lookup
    obs_set = set(int(x) for x in observed_indices)
    obs_map = {int(observed_indices[k]): k for k in range(K)}
    
    # Adaptive multiplicative inflation
    if M > 2:
        spread = np.std(ensemble[:, list(obs_set)], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(spread) / obs_err if obs_err > 0 else 1.0
        if ratio < 1.0:
            inflation = 1.0 + 0.03 * (1.0 - ratio)
        else:
            inflation = 1.002
        inflation = float(np.clip(inflation, 1.002, 1.03))
    else:
        inflation = 1.002
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Correlation length for localization
    corr_len = 4.0
    
    # Precompute localization weights for each center i
    loc_weights_by_i = []
    for i in range(N):
        dists = np.array([_periodic_distance(i, (i + j) % N, N) for j in range(-3, 4)])
        w = _gaspari_cohn(dists, corr_len)
        w[3] = 1.0  # center weight
        loc_weights_by_i.append(w)
    
    # Time budget
    time_budget = 145.0
    
    for i in range(N):
        if time.time() - t0 > time_budget:
            break
        
        # Local window indices (periodic)
        local_idx = [(i + j) % N for j in range(-3, 4)]
        
        # Find which local positions are observed
        local_obs_pos = []
        for l, g in enumerate(local_idx):
            if g in obs_set:
                local_obs_pos.append(l)
        
        if len(local_obs_pos) == 0:
            continue
        
        n_obs_local = len(local_obs_pos)
        n_local = len(local_idx)  # 7
        
        # Extract local ensemble: (M, 7)
        local_ens = result[:, local_idx].copy()
        
        # Local observations
        y_local = np.array([observations[obs_map[local_idx[l]]] for l in local_obs_pos])
        
        # Local mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean  # (M, 7)
        
        # Apply localization to anomalies (element-wise)
        w = loc_weights_by_i[i]
        local_anoms_loc = local_anoms * w[np.newaxis, :]
        
        # Local ensemble covariance with localization: C = (1/(M-1)) * A_loc^T A_loc
        C = (local_anoms_loc.T @ local_anoms_loc) / (M - 1)  # (7, 7)
        
        # Observation operator: (n_obs_local, 7)
        H = np.zeros((n_obs_local, n_local))
        for k, l in enumerate(local_obs_pos):
            H[k, l] = 1.0
        
        # Innovation covariance S = H C H^T + R
        HC = H @ C  # (n_obs_local, 7)
        S = HC @ H.T + observation_variance * np.eye(n_obs_local)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        # Posterior covariance: C_post = C - C H^T S^{-1} H C
        CHtS = C @ H.T @ S_inv  # (7, n_obs_local)
        C_post = C - CHtS @ H @ C  # (7, 7)
        
        # Ensure positive semi-definiteness
        eigvals, eigvecs = np.linalg.eigh(C_post)
        eigvals = np.maximum(eigvals, 0)
        C_post = (eigvecs * eigvals) @ eigvecs.T
        C_post = 0.5 * (C_post + C_post.T)
        
        # Compute mean update
        Hx_mean = H @ local_mean
        innov_mean = y_local - Hx_mean
        K_mean = C @ H.T @ S_inv
        local_mean_new = local_mean + K_mean @ innov_mean
        
        # Compute scale factors to match posterior variance
        var_f = np.diag(C)
        var_a = np.diag(C_post)
        scale_factors = np.ones(n_local)
        for j in range(n_local):
            if var_f[j] > 1e-10:
                scale_factors[j] = np.sqrt(max(var_a[j] / var_f[j], 0.0))
        
        # Apply the scale factors to the anomalies
        local_anoms_new = local_anoms * scale_factors[np.newaxis, :]
        
        # Update the local ensemble for all members
        for m in range(M):
            result[m, local_idx] = local_mean_new + local_anoms_new[m, :]
    
    # Safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
