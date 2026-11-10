"""Lorenz-96 ensemble assimilation with local EnKF and adaptive inflation."""
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


def _local_enkf_update(ensemble, observations, observed_indices, observation_variance,
                       n_members=16, n_state=20, n_obs=10,
                       localization_scale=4.0, inflation=1.05):
    """Local EnKF update with adaptive inflation and Gaspari-Cohn localization."""
    result = ensemble.copy().astype(float)
    
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    
    # Adaptive inflation based on innovation statistics
    innovations = observations - mean[observed_indices]
    ens_spread_obs = np.std(perturbations[:, observed_indices], axis=0)
    ratio = np.mean(np.abs(innovations) / (ens_spread_obs + 1e-10))
    
    if ratio > 1.15:
        inflation_factor = 1.0 + 0.2 * (ratio - 1.15)
        inflation_factor = min(inflation_factor, 1.5)
    else:
        inflation_factor = inflation
    
    perturbations = perturbations * inflation_factor
    
    # Precompute localization weights
    loc_weights = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j, obs_idx in enumerate(observed_indices):
            dist = _periodic_distance(i, obs_idx, n_state)
            loc_weights[i, j] = _gaspari_cohn(np.array([dist]), c=localization_scale)[0]
    
    # For each state variable, perform local EnKF update
    for i in range(n_state):
        x_i_pert = perturbations[:, i]
        
        w = loc_weights[i, :]
        valid = w > 0.05
        if not np.any(valid):
            result[:, i] = mean[i] + x_i_pert
            continue
        
        obs_sel = observations[valid]
        w_sel = w[valid]
        n_valid = np.sum(valid)
        
        P_f = np.var(x_i_pert, ddof=1) if n_members > 1 else 0.0
        
        R = observation_variance / w_sel
        H = w_sel
        
        S = np.outer(H, H) * P_f + np.diag(R)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            result[:, i] = mean[i] + x_i_pert
            continue
        
        K = P_f * H @ S_inv
        K_H = float(np.dot(K, H))
        
        innovation = obs_sel - np.dot(H, mean[observed_indices[valid]])
        K_innov = float(np.dot(K, innovation))
        
        for m in range(n_members):
            result[m, i] = mean[i] + x_i_pert[m] * (1.0 - K_H) + K_innov
    
    # Clip to reasonable bounds
    result = np.clip(result, -100.0, 100.0)
    result[~np.isfinite(result)] = 0.0
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """Perform local EnKF analysis update."""
    t_start = time.time()
    deadline = t_start + 110.0
    
    try:
        n_members, n_state = ensemble.shape
        n_obs = len(observed_indices)
        
        result = _local_enkf_update(
            ensemble, observations, observed_indices, observation_variance,
            n_members=n_members, n_state=n_state, n_obs=n_obs,
            localization_scale=4.0, inflation=1.05
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
