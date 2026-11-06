"""Local Sequential Ensemble Kalman Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Sequential Ensemble Kalman Filter (EnKF) analysis update for Lorenz-96.
    
    This implementation processes observations sequentially, updating only a local 
    neighborhood of state variables for each observation. This structural choice 
    avoids the spurious long-range correlations inherent in global additive updates 
    and is consistent with the Local Ensemble Transform Kalman Filter (LETKF) 
    philosophy, which has shown superior performance for this chaotic system.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)  # 10
    obs_var = float(observation_variance)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    
    # --- Global Adaptive Inflation ---
    # Estimate if the ensemble spread is too small relative to the observations.
    # We use a simple heuristic: compare the variance of the observations with the 
    # ensemble variance at the observed locations.
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Compute expected innovation variance from ensemble
    # H is identity for observed indices
    H_ens_anoms = anomalies[:, observed_indices]  # (nm, nobs)
    ens_obs_var = np.var(H_ens_anoms, axis=0, ddof=1)  # (nobs,)
    
    # Observed innovation variance (approximated by variance of obs - mean_ens)
    innovations_global = observations - ens_mean[observed_indices]
    obs_innov_var = np.var(innovations_global, ddof=1)
    
    # Theoretical innovation variance is ens_spread + obs_noise
    # We want the inflated spread to match the observed spread roughly.
    # Simple ratio-based inflation
    mean_ens_var = np.mean(ens_obs_var)
    if mean_ens_var > 1e-6:
        # Ratio of observed innovation variance to ensemble variance
        # If obs spread is larger than ensemble spread, we are under-dispersive.
        ratio = obs_innov_var / (mean_ens_var + obs_var)
        # Inflation factor alpha: scale anomalies by sqrt(alpha)
        # We want new_var * alpha + obs_var ~ obs_innov_var
        # So alpha ~ (obs_innov_var - obs_var) / mean_ens_var
        target_alpha = (obs_innov_var - obs_var) / mean_ens_var
        if target_alpha < 1.0:
            target_alpha = 1.0
        # Cap inflation to prevent explosion
        inflation_factor = np.clip(target_alpha, 1.0, 2.0)
    else:
        inflation_factor = 1.5  # Default strong inflation if spread is zero
    
    # Apply inflation
    inflated_ensemble = ens_mean + np.sqrt(inflation_factor) * anomalies
    
    # --- Local Sequential Update ---
    # Process each observation one by one
    # Local window radius
    local_radius = 4
    
    # Pre-compute local indices for each observation
    # observed_indices are even indices: 0, 2, ..., 18
    updated_ensemble = inflated_ensemble.copy()
    
    for k, obs_idx in enumerate(observed_indices):
        obs_val = observations[k]
        obs_idx_int = int(obs_idx)
        
        # Determine local window
        # Indices: obs_idx - local_radius to obs_idx + local_radius
        # Handle periodic boundary (mod nvars)
        local_indices = []
        for d in range(-local_radius, local_radius + 1):
            local_indices.append((obs_idx_int + d) % nvars)
        
        # Unique local indices
        local_indices = np.array(list(set(local_indices)))
        
        # Extract local ensemble
        local_ens = updated_ensemble[:, local_indices]  # (nm, n_local)
        local_mean = np.mean(local_ens, axis=0)
        local_anoms = local_ens - local_mean
        
        # Local covariance
        n_local = len(local_indices)
        # C = (1/(nm-1)) * A^T A
        C_local = (local_anoms.T @ local_anoms) / (nm - 1)  # (n_local, n_local)
        
        # Observation operator for this local step:
        # We are observing one variable (obs_idx_int) which is at position 'center' in local_indices
        # Find the position of obs_idx_int in local_indices
        try:
            center_pos = np.where(local_indices == obs_idx_int)[0][0]
        except IndexError:
            # Should not happen, but safety
            continue
            
        # H_local is a row vector with 1 at center_pos
        H_local = np.zeros(n_local)
        H_local[center_pos] = 1.0
        
        # Innovation
        innovation = obs_val - local_mean[center_pos]
        
        # Observation variance
        R = obs_var
        
        # Calculate Kalman Gain for this local step
        # P_oo = H C H^T + R
        P_oo = H_local @ C_local @ H_local + R
        
        # K = C H^T P_oo^{-1}
        # K is (n_local, 1)
        K_local = (C_local @ H_local) / P_oo
        
        # Update local ensemble members
        for m in range(nm):
            # x_f = local_mean + local_anoms[m]
            # x_a = x_f + K * (y - H x_f)
            # H x_f = local_mean[center_pos] + local_anoms[m, center_pos]
            H_x_f = local_mean[center_pos] + local_anoms[m, center_pos]
            innov_m = obs_val - H_x_f
            update_vec = K_local * innov_m
            local_ens[m] = local_ens[m] + update_vec
            
        # Write back to global ensemble
        updated_ensemble[:, local_indices] = local_ens
        
        # Deadline check
        if time.time() - t0 > 115.0:
            break
            
    # --- Final Safety Checks ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
