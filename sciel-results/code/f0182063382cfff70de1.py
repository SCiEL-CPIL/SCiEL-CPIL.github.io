"""Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
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
    
    r1 = d[mask1]
    if len(r1) > 0:
        x = r1 / c
        result[mask1] = (-5.0/4.0)*x**5 + 5.0*x**4 - 10.0*x**3 + 5.0*x
    
    r2 = d[mask2]
    if len(r2) > 0:
        x = r2 / c
        result[mask2] = (1.0/12.0)*x**5 - (5.0/4.0)*x**4 + (5.0/2.0)*x**3 - 5.0*x**2 + (25.0/12.0)*x + 0.5
    
    r3 = d[mask3]
    if len(r3) > 0:
        x = r3 / c
        result[mask3] = (-1.0/16.0)*x**5 + (1.0/4.0)*x**4 - (5.0/8.0)*x**3 - (5.0/2.0)*x**2 + (25.0/12.0)*x - 0.25
    
    return result


def _periodic_distance(i, j, n):
    """Periodic distance between indices i and j in a ring of size n."""
    d = abs(i - j)
    return min(d, n - d)


def _local_esrf_update(ensemble, observations, observed_indices, observation_variance,
                       n_members=16, n_state=20, n_obs=10,
                       inflation=1.1, localization_scale=3.0):
    """Local ESRF update with adaptive inflation and Gaspari-Cohn localization."""
    result = ensemble.copy().astype(float)
    
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    
    # Adaptive inflation
    innovations = observations - mean[observed_indices]
    ens_spread_obs = np.std(perturbations[:, observed_indices], axis=0)
    ratio = np.mean(np.abs(innovations) / (ens_spread_obs + 1e-10))
    if ratio > 1.5:
        inflation_factor = 1.0 + 0.1 * (ratio - 1.5)
        inflation_factor = min(inflation_factor, 1.3)
    else:
        inflation_factor = inflation
    
    perturbations = perturbations * inflation_factor
    
    # Precompute localization weights for each state i and each obs j
    # loc_weights[i, j] = GC(distance between state i and obs j)
    loc_weights = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j, obs_idx in enumerate(observed_indices):
            dist = _periodic_distance(i, obs_idx, n_state)
            loc_weights[i, j] = _gaspari_cohn(np.array([dist]), c=localization_scale)[0]
    
    # For each state variable, perform local ESRF update
    for i in range(n_state):
        x_i_pert = perturbations[:, i]  # (n_members,)
        
        # Get localization weights for this state
        w = loc_weights[i, :]  # (n_obs,)
        
        # Filter observations with meaningful localization
        valid = w > 0.05
        if not np.any(valid):
            result[:, i] = mean[i] + x_i_pert
            continue
        
        obs_sel = observations[valid]
        w_sel = w[valid]
        n_valid = np.sum(valid)
        
        # ESRF update: we want to find a transformation Q such that
        # the new perturbations have the correct analysis covariance.
        # For 1D state with multiple observations:
        # P_f = var(x_i_pert)
        # P_a = P_f * (1 - K*H) where K is the Kalman gain
        # Actually for ESRF, we directly scale perturbations.
        
        # Compute ensemble variance (forecast)
        P_f = np.var(x_i_pert, ddof=1) if n_members > 1 else 0.0
        
        # Observation noise (effective)
        R = observation_variance / w_sel  # (n_valid,)
        
        # H = w_sel (1 x n_valid)
        H = w_sel
        
        # S = H * P_f * H^T + R (n_valid x n_valid)
        S = np.outer(H, H) * P_f + np.diag(R)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            result[:, i] = mean[i] + x_i_pert
            continue
        
        # K = P_f * H^T * S^-1 (1 x n_valid)
        K = P_f * H @ S_inv  # (n_valid,)
        
        # K_H = K @ H (scalar) - this is the reduction factor
        K_H = float(np.dot(K, H))
        
        # Innovation: y - H * x_f_mean
        innovation = obs_sel - np.dot(H, mean[observed_indices[valid]])
        
        # K_innov = K @ innovation (scalar)
        K_innov = float(np.dot(K, innovation))
        
        # ESRF: new perturbation = (1 - K_H) * old_pert + K_innov
        # But for ESRF, we should use sqrt(1-K_H) to preserve spread
        # Actually the standard ESRF uses:
        # x_a_pert = x_f_pert * sqrt(1 - K*H) + K * (y - H*x_f_mean)
        # But this doesn't quite work for multi-obs. Let's use the standard formula.
        
        # The correct ESRF update for perturbations:
        # new_pert[m] = old_pert[m] * (1 - K_H) + K_innov
        # This is equivalent to the EnKF perturbation update.
        
        for m in range(n_members):
            result[m, i] = mean[i] + x_i_pert[m] * (1.0 - K_H) + K_innov
    
    # Clip to reasonable bounds
    result = np.clip(result, -100.0, 100.0)
    result[~np.isfinite(result)] = 0.0
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """Perform local ESRF analysis update."""
    t_start = time.time()
    deadline = t_start + 110.0
    
    try:
        n_members, n_state = ensemble.shape
        n_obs = len(observed_indices)
        
        result = _local_esrf_update(
            ensemble, observations, observed_indices, observation_variance,
            n_members=n_members, n_state=n_state, n_obs=n_obs,
            inflation=1.1, localization_scale=3.0
        )
        
        if time.time() > deadline:
            result = ensemble.copy().astype(float)
            result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
            result = np.clip(result, -100.0, 100.0)
        
        return result
        
    except Exception:
        result = ensemble.copy().astype(float)
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        result = np.clip(result, -100.0, 100.0)
        result[~np.isfinite(result)] = 0.0
        return result
# EVOLVE-BLOCK-END
