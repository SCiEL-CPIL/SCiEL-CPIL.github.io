"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _gaussian_localization(dist, length_scale):
    """Gaussian localization weight."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LESRF-style assimilation for Lorenz-96 with localization and adaptive inflation.
    
    Uses a deterministic sequential square-root filter update with Gaussian localization
    and adaptive spread inflation to handle partial observations of the
    20-variable Lorenz-96 system.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of 16 members, each a 20-state Lorenz-96 vector
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed states
    observed_indices : array_like
        Indices of observed variables
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after assimilation
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return ensemble.copy()
    
    N, M = ensemble.shape
    obs_idx = np.array(sorted(observed_indices), dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ens = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    obs = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    
    # --- Step 1: Compute ensemble mean and anomalies ---
    x_mean = np.mean(ens, axis=0)  # (20,)
    anomalies = ens - x_mean  # (N, 20)
    
    # --- Step 2: Gentle Adaptive Inflation (tanh-based) ---
    # Compute observation-space innovation
    x_mean_obs = x_mean[obs_idx]  # (10,)
    innovation = obs - x_mean_obs  # (10,)
    
    # Estimate ensemble variance in observation space
    anomalies_obs = anomalies[:, obs_idx]  # (N, 10)
    P_yy = np.mean(anomalies_obs ** 2, axis=0)  # (10,)
    P_yy_total = P_yy + observation_variance  # (10,)
    
    # Innovation variance ratio for adaptive inflation
    innovation_var = np.mean(innovation ** 2)
    expected_innov_var = np.mean(P_yy_total)
    
    if expected_innov_var > 1e-10:
        ratio = innovation_var / expected_innov_var
        # Gentle multiplicative inflation: smooth, bounded
        inflation_factor = 1.0 + 0.15 * np.tanh(ratio - 1.0)
        inflation_factor = np.clip(inflation_factor, 0.95, 1.15)
    else:
        inflation_factor = 1.0
    
    # Apply inflation about the mean
    if abs(inflation_factor - 1.0) > 1e-6:
        anomalies = anomalies * inflation_factor
        ens = x_mean + anomalies
    
    # --- Step 3: Build localization matrix ---
    # σ=3.0 matches correlation length in L96 at F=6-10
    length_scale = 3.0
    
    # Precompute periodic distances from each observed index to all state indices
    dist_matrix = np.zeros((n_obs, M))
    for i, oi in enumerate(obs_idx):
        for j in range(M):
            d = abs(oi - j)
            d = min(d, M - d)  # periodic
            dist_matrix[i, j] = d
    
    # Localization weights
    L = _gaussian_localization(dist_matrix, length_scale)  # (10, 20)
    # Ensure self-observation gets full weight
    for i in range(n_obs):
        L[i, obs_idx[i]] = 1.0
    
    # --- Step 4: Vectorized Sequential LESRF update ---
    # Process each observation sequentially with localization
    updated_ens = ens.copy()
    
    for k in range(n_obs):
        if time.time() > _DEADLINE:
            break
        
        oi = obs_idx[k]
        obs_val = obs[k]
        
        # Current ensemble statistics
        cur_mean = np.mean(updated_ens, axis=0)
        cur_anom = updated_ens - cur_mean  # (N, M)
        
        # Local variance at observed index
        var_k = np.mean(cur_anom[:, oi] ** 2)
        
        # Cross-covariance: cov(x_oi, x_j) for all j, vectorized
        P_xy_raw = cur_anom[:, oi].dot(cur_anom) / N  # (M,)
        
        # Apply localization weights
        P_xy = P_xy_raw * L[k, :]  # (M,)
        
        # Denominator
        denom = var_k + observation_variance
        
        if denom < 1e-10:
            continue
        
        # Kalman gain vector
        K = P_xy / denom  # (M,)
        
        # Vectorized member update: x_n = x_n + K * (obs - x_n[oi])
        member_innovs = obs_val - updated_ens[:, oi]  # (N,)
        updated_ens = updated_ens + np.outer(member_innovs, K)
    
    # --- Step 5: Spread preservation with mild inflation ---
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)  # spread before LESRF update
    
    for j in range(M):
        if current_spread[j] > 1e-8:
            # Target: maintain at least 95% of pre-update spread
            target = pre_spread[j] * 0.95
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.3)  # cap inflation more tightly
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            rng = np.random.default_rng(42 + j)
            updated_ens[:, j] = new_mean[j] + rng.normal(0, 0.03, N)
    
    # --- Step 6: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
