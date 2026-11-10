"""Global ESRF with Gaspari-Cohn localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn_matrix(N, c):
    """Build N x N Gaspari-Cohn localization matrix with periodic distance."""
    d = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    d = np.minimum(d, N - d)  # periodic distance
    r = d / c
    result = np.zeros_like(r)
    
    # Piecewise polynomial
    # r <= 0.5
    m1 = r <= 0.5
    r1 = r[m1]
    result[m1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
    
    # 0.5 < r <= 1.5
    m2 = (r > 0.5) & (r <= 1.5)
    r2 = r[m2]
    result[m2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
    
    # 1.5 < r <= 2.5
    m3 = (r > 1.5) & (r <= 2.5)
    r3 = r[m3]
    result[m3] = (-1.0/16.0)*r3**5 + (1.0/4.0)*r3**4 - (15.0/8.0)*r3**3 + (5.0/2.0)*r3**2 - (25.0/12.0)*r3 + 105.0/48.0
    
    return result

def _global_esrf_update(ens, obs_idx, obs_val, obs_var, N, loc_width):
    """
    Perform a global ESRF update with localization.
    """
    M, _ = ens.shape
    
    # 1. Compute background statistics
    mean = ens.mean(axis=0)
    anom = ens - mean
    
    # 2. Compute localized background covariance
    # C_bg = (anom.T @ anom) / (M-1)
    C_bg = (anom.T @ anom) / (M - 1)
    
    # Apply localization
    L = _gaspari_cohn_matrix(N, loc_width)
    C_loc = C_bg * L
    
    # 3. Observation operator H is a selection matrix
    # H C_loc H^T is the covariance of observations
    # Since H just selects rows, H C_loc H^T is the submatrix of C_loc
    # at indices obs_idx
    C_obs = C_loc[np.ix_(obs_idx, obs_idx)]
    
    # 4. Innovation covariance S = H C_loc H^T + R
    R = obs_var * np.eye(len(obs_idx))
    S = C_obs + R
    
    # 5. Gain K = C_loc H^T S^-1
    # C_loc H^T is the submatrix of C_loc with columns obs_idx
    C_loc_Ht = C_loc[:, obs_idx]
    
    # Solve for Gain
    # K = C_loc_Ht @ inv(S)
    # To be stable, use solve
    try:
        # S is small (10x10 for L96 half obs)
        S_inv = np.linalg.inv(S)
        K = C_loc_Ht @ S_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        S_inv = np.linalg.pinv(S)
        K = C_loc_Ht @ S_inv
        
    # 6. Update ensemble
    # Innovation y - H x
    innovations = obs_val - ens[:, obs_idx] # M x K_obs
    
    # Update: x_new = x + K @ innovations.T
    # K is N x K_obs, innovations.T is K_obs x M
    # Result is N x M
    ens_new = ens + (K @ innovations.T).T
    
    return ens_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Global ESRF with adaptive inflation for Lorenz-96.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # 1. Adaptive Inflation
    obs_std = np.sqrt(observation_variance)
    ens_spread = np.std(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    
    ratio = mean_spread / obs_std if obs_std > 0 else 1.0
    
    # Tuned inflation logic
    if ratio < 1.0:
        inflation = 1.0 + 0.3 * (1.0 - ratio)
    else:
        inflation = 1.0 + 0.1 * np.exp(-(ratio - 1.0))
        
    inflation = np.clip(inflation, 1.0, 1.8)
    
    # Apply multiplicative inflation
    mean = ensemble.mean(axis=0)
    inflated = mean + inflation * (ensemble - mean)
    
    # 2. Global ESRF Update
    # Localization width: typically 2-4 grid points for L96
    # With N=20, a width of 3-4 is reasonable.
    loc_width = 4.0 
    
    updated = _global_esrf_update(inflated, observed_indices, observations, observation_variance, N, loc_width)
    
    # 3. Safety Checks
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
