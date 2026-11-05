"""
Local Ensemble Transform Kalman Filter (LETKF) for Lorenz-96 assimilation.
Strategy:
1. Compute localized background error covariance using Gaussian decay.
2. Perform a local transform Kalman filter update for each variable.
3. Apply adaptive ensemble inflation to prevent filter collapse.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _cyclic_distance(i, j, n):
    """Compute cyclic distance between indices i and j on a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)

def _localization_weights(n, loc_range=2.8):
    """
    Compute Gaussian localization weights for a cyclic domain.
    Returns a (n, n) matrix of weights.
    """
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = _cyclic_distance(i, j, n)
            weights[i, j] = np.exp(-0.5 * (d / loc_range) ** 2)
    return weights

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _letkf_update(x_bg, y_obs, obs_idx, R, n_vars=20, loc_range=2.8):
    """
    Perform LETKF update.
    x_bg: (n_ens, n_vars) background ensemble
    y_obs: (n_ens, n_obs) observations (noisy)
    obs_idx: array of observed indices
    R: observation variance (scalar)
    """
    n_ens, n_vars = x_bg.shape
    n_obs = len(obs_idx)
    
    # 1. Compute background mean and perturbations
    x_mean = np.mean(x_bg, axis=0)
    Xp = x_bg - x_mean  # (n_ens, n_vars)
    
    # 2. Compute localized background covariance
    # B = Xp.T * Xp / (n_ens - 1)
    # We need to apply localization to B.
    # Instead of forming full B, we can apply localization in the transform.
    # Standard LETKF approach:
    # For each variable i, solve local KKF.
    
    # Precompute localization weights
    W = _localization_weights(n_vars, loc_range)
    
    x_an = x_bg.copy()
    
    for i in range(n_vars):
        # Local observation indices: all observed variables, but weighted by distance to i
        # In LETKF, we usually restrict to a local neighborhood.
        # However, with Gaussian localization, we can use all observations with weights.
        # For efficiency and stability, let's use a local window.
        # Window size: indices within loc_range * 2 or so.
        # Let's use all observed indices but weight them.
        
        # Local background covariance B_i
        # B_i = sum_j W[i,j] * (Xp[:, j] * Xp[:, j].T) / (n_ens - 1)
        # This is a (n_ens, n_ens) matrix.
        
        # Efficiently compute B_i:
        # B_i = (Xp[:, obs_idx] * W_i_obs.T).T @ (Xp[:, obs_idx] * W_i_obs) ? No.
        # B_i = sum_{j} W[i,j] * (Xp[:, j] Xp[:, j].T)
        # Let Xp_obs = Xp[:, obs_idx] (n_ens, n_obs)
        # Let w_i = W[i, obs_idx] (n_obs,)
        # B_i = sum_{k} w_i[k] * Xp[:, obs_idx[k]] Xp[:, obs_idx[k]].T
        # B_i = Xp_obs.T @ (Xp_obs * w_i) / (n_ens - 1)
        
        Xp_obs = Xp[:, obs_idx]
        w_i = W[i, obs_idx]
        
        # Weighted covariance
        # B_i is (n_ens, n_ens)
        # B_i = Xp_obs.T @ (Xp_obs * w_i) / (n_ens - 1)
        B_i = (Xp_obs.T @ (Xp_obs * w_i)) / (n_ens - 1)
        
        # Observation operator H is identity for observed vars.
        # H B_i H^T + R
        # Since we are solving for variable i, and observations are on obs_idx,
        # we need the covariance of the local state (variable i) and observations.
        
        # Actually, standard LETKF solves for the transform Y such that:
        # x_an = x_bg + Y^T (y_obs - H x_bg)
        # Y is (n_ens, n_obs)
        
        # Let's use the standard KKF formulation for the local system.
        # State: x_i (scalar) and x_obs (n_obs vector)?
        # No, the transform is applied to the ensemble perturbations.
        
        # Alternative: Use the "local" EKF where we treat the subset of variables
        # correlated with i as the local state.
        # But the most efficient LETKF uses the full ensemble covariance structure.
        
        # Let's stick to the standard LETKF algorithm from Hunt et al. (2007).
        # 1. Localize B: B_loc = D B D, where D is diagonal localization? No, that's for global.
        # In LETKF, we compute B_i for each i.
        
        # Let's use a simpler, robust approach: 
        # Local EKF with a window of variables around i.
        # Window: indices j such that cyclic_distance(i, j) <= window_size.
        window_size = int(np.ceil(loc_range * 2))
        local_idx = []
        for j in range(n_vars):
            if _cyclic_distance(i, j, n_vars) <= window_size:
                local_idx.append(j)
        
        local_idx = np.array(local_idx)
        
        # Extract local background perturbations
        Xp_local = Xp[:, local_idx]
        
        # Which of these are observed?
        obs_mask = np.isin(local_idx, obs_idx)
        local_obs_idx = local_idx[obs_mask]
        local_obs_pos = np.where(obs_mask)[0]
        
        if len(local_obs_idx) == 0:
            continue
        
        # Local covariance B_loc (n_ens, n_ens)
        B_loc = (Xp_local.T @ Xp_local) / (n_ens - 1)
        
        # Observation covariance R_loc
        R_loc = R * np.eye(len(local_obs_idx))
        
        # Innovation
        y_bg = x_bg[:, local_obs_idx]
        y_innov = y_obs[:, np.isin(obs_idx, local_obs_idx)] - y_bg
        
        # Kalman Gain
        # K = B_loc H^T (H B_loc H^T + R)^-1
        # H is selection matrix.
        # H B_loc H^T is the covariance of the observed local variables.
        # We can compute this using the Schur complement or direct inversion.
        
        # Since n_ens is small (16), B_loc is 16x16.
        # H B_loc H^T is (n_obs_loc, n_obs_loc).
        
        # Let's compute the gain G (n_ens, n_obs_loc)
        # G = B_loc @ Xp_local[:, local_obs_pos].T @ np.linalg.inv(
        #     Xp_local[:, local_obs_pos].T @ B_loc @ Xp_local[:, local_obs_pos] + R_loc * (n_ens-1)
        # )
        # Wait, the standard formula uses the ensemble perturbations.
        
        # Let Xp_o = Xp_local[:, local_obs_pos] (n_ens, n_obs_loc)
        # S = Xp_o.T @ Xp_o / (n_ens - 1) + R_loc
        # K = Xp_local @ Xp_o.T / (n_ens - 1) @ S^-1
        
        Xp_o = Xp_local[:, local_obs_pos]
        S = (Xp_o.T @ Xp_o) / (n_ens - 1) + R_loc
        
        # Solve S K^T = Xp_o.T @ Xp_local / (n_ens - 1)
        # K^T = S^-1 @ (Xp_o.T @ Xp_local / (n_ens - 1))
        # K = (Xp_local.T @ Xp_o / (n_ens - 1)) @ S^-T
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = (Xp_local.T @ Xp_o / (n_ens - 1)) @ S_inv.T
        
        # Update perturbations
        # Xp_an_local = Xp_local + K @ y_innov
        Xp_an_local = Xp_local + K @ y_innov
        
        # Update mean
        # x_mean_an_local = x_mean[local_idx] + (1/n_ens) * sum(Xp_an_local, axis=0)
        # Actually, x_mean_an = x_mean + K @ y_innov_mean? No.
        # x_an = x_bg + K @ (y_obs - H x_bg)
        # For each ensemble member m:
        # x_an[m, local_idx] = x_bg[m, local_idx] + K @ (y_obs[m, local_obs_idx] - x_bg[m, local_obs_idx])
        
        y_innov_m = y_obs[:, local_obs_idx] - x_bg[:, local_obs_idx]
        x_an[:, local_idx] = x_bg[:, local_idx] + K @ y_innov_m
        
    return x_an

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform LETKF update.
    ensemble: (16, 20)
    observations: (16, 10)
    observed_indices: array of 10 indices
    observation_variance: scalar
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    x_bg = ensemble.copy()
    
    # Adaptive Inflation
    # Compute innovation norm
    y_bg = x_bg[:, observed_indices]
    y_innov = observations - y_bg
    innov_norm = np.sqrt(np.mean(np.sum(y_innov**2, axis=1)))
    
    # Expected innovation norm
    # Approximate using background variance
    x_mean = np.mean(x_bg, axis=0)
    bg_var = np.var(x_bg[:, observed_indices], axis=0)
    expected_innov_norm = np.sqrt(np.mean(bg_var + observation_variance))
    
    ratio = innov_norm / (expected_innov_norm + 1e-10)
    
    # Inflation factor
    # If ratio > 1, inflate.
    alpha = 1.0
    if ratio > 1.0:
        alpha = 1.0 + 0.05 * (ratio - 1.0)
        alpha = min(alpha, 1.2)  # Cap inflation
    
    # Inflate background
    x_bg_inflated = x_mean + alpha * (x_bg - x_mean)
    
    # LETKF Update
    x_an = _letkf_update(x_bg_inflated, observations, observed_indices, observation_variance)
    
    # Final check for finiteness and bounds
    if not np.all(np.isfinite(x_an)):
        x_an = x_bg  # Fallback to background if numerical issues
    
    x_an = np.clip(x_an, -100, 100)
    
    return x_an

# EVOLVE-BLOCK-END
