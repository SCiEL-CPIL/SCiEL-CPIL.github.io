"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 5s safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _gaussian_localization(dist, length_scale):
    """Gaussian localization weight."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LESRF-style assimilation for Lorenz-96 with localization and adaptive inflation.
    
    Uses a sequential square-root filter update with Gaussian localization
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
    
    # --- Step 2: Adaptive Inflation ---
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
        # Smooth adaptive inflation using tanh-based soft clamp
        # soft_clamp(x, lo, hi, alpha) = mid + alpha * tanh((x - mid) / alpha)
        # where mid = (lo+hi)/2, but we want asymmetric behavior
        # Use: if ratio in [0.7, 2.5], pass through; smooth outside
        lo, hi = 0.7, 2.5
        alpha = 0.3
        # Smooth lower bound: for x < lo, smoothly approaches lo
        # Smooth upper bound: for x > hi, smoothly approaches hi
        # Combined: use sigmoid blending
        w_lo = 1.0 / (1.0 + np.exp((ratio - lo) / alpha))
        w_hi = 1.0 / (1.0 + np.exp((hi - ratio) / alpha))
        # In the middle, both weights ~0.5, so blend toward ratio
        # At extremes, one weight dominates
        smooth_ratio = (1.0 - w_lo - w_hi) * ratio + w_lo * lo + w_hi * hi
        # Clamp to safe range
        smooth_ratio = np.clip(smooth_ratio, 0.5, 3.0)
        inflation_factor = np.sqrt(smooth_ratio)
    else:
        inflation_factor = 1.0
    
    # Apply inflation about the mean
    if abs(inflation_factor - 1.0) > 1e-6:
        anomalies = anomalies * inflation_factor
        ens = x_mean + anomalies
    
    # --- Step 3: Build localization matrix ---
    # Localization scale tuned for L96 correlation length (~3-4 grid points)
    length_scale = 2.8
    
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
    
    # --- Step 4: Sequential LESRF update (vectorized) ---
    # Process observations in a shuffled order to reduce sequential bias
    # Use a deterministic pseudo-random shuffle based on observation values
    # for reproducibility
    updated_ens = ens.copy()
    
    # Create a deterministic ordering: sort by observation index + small perturbation
    # to break ties and reduce sequential correlation bias
    order = np.argsort(obs_idx + 0.1 * np.sin(np.arange(n_obs) * 7.13))
    
    for k in range(n_obs):
        if time.time() > _DEADLINE:
            break
        
        oi = obs_idx[order[k]]
        obs_val = obs[k]
        
        # Current ensemble statistics
        cur_mean = np.mean(updated_ens, axis=0)  # (M,)
        cur_anom = updated_ens - cur_mean  # (N, M)
        
        # Local variance at observation point
        x_k_anom = cur_anom[:, oi]  # (N,)
        var_k = np.mean(x_k_anom ** 2)
        
        # Denominator
        denom = var_k + observation_variance
        
        if denom < 1e-10:
            continue
        
        # Cross-covariance vector: cov(x_k, x_j) for all j, with localization
        # P_xy_j = (1/N) * sum_n (cur_anom[n, oi] * cur_anom[n, j]) * L[k, j]
        # Vectorized: (N,M) @ (N,) gives (M,), then scale by L[k,:]
        P_xy = (cur_anom.T @ x_k_anom) / N * L[order[k], :]  # (M,)
        
        # Kalman gain vector
        K = P_xy / denom  # (M,)
        
        # Innovation for each member: obs_val - updated_ens[n, oi]
        member_innov = obs_val - updated_ens[:, oi]  # (N,)
        
        # Update all members at once: updated_ens[n, :] += K * member_innov[n]
        updated_ens += np.outer(member_innov, K)
    
    # --- Step 5: Variance inflation to prevent collapse ---
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)  # spread before LESRF update
    
    # Per-component inflation with more aggressive spread restoration
    for j in range(M):
        if current_spread[j] > 1e-8:
            # Target: maintain at least 90% of pre-update spread
            target = pre_spread[j] * 0.90
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.5)  # allow more inflation to restore spread
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            updated_ens[:, j] = new_mean[j] + np.random.normal(0, 0.03, N)
    
    # --- Step 6: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
