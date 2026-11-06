"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for an array of states (batch)."""
    n_vars = x.shape[-1]
    x_shifted1 = np.roll(x, -1, axis=-1)  # x_{i+1}
    x_shifted2 = np.roll(x, 2, axis=-1)   # x_{i-2}
    x_shifted3 = np.roll(x, 1, axis=-1)   # x_{i-1}
    return (x_shifted1 - x_shifted2) * x_shifted3 - x + F

def _gaspari_cohn(dist, radius):
    """Gaspari-Cohn tapering function for localization."""
    if dist >= radius:
        return 0.0
    r = dist / radius
    if r <= 0.5:
        return 1.0 - 1.25 * r**2 + 0.625 * r**4
    else:
        return 2.5 - 1.25 * r**2 - 2.5 * r**3 + 0.25 * r**4

def _letkf_transform_single_obs(anom, obs_pos, obs_var_ens, k):
    """
    Compute the LETKF-transformed anomalies for a single observation.
    
    Parameters:
    - anom: (n_members, n_local) anomaly matrix
    - obs_pos: position of observed variable in the local block
    - obs_var_ens: ensemble variance of the observed variable
    - k: Kalman gain = obs_var_ens / (obs_var_ens + obs_var)
    
    Returns:
    - new_anom: (n_members, n_local) transformed anomaly matrix
    """
    n_members, n_local = anom.shape
    a = anom[:, obs_pos].copy()
    new_anom = anom.copy()
    
    if obs_var_ens < 1e-12:
        return new_anom
    
    # Scale the observed variable by (1-k) to reduce its variance
    new_anom[:, obs_pos] = (1.0 - k) * a
    
    # Update other variables based on their correlation with the observed variable
    for j in range(n_local):
        if j == obs_pos:
            continue
        # Covariance between variable j and the observed variable
        cov_jp = np.dot(anom[:, j], a) / (n_members - 1)
        # LETKF update: add correlated innovation
        new_anom[:, j] = anom[:, j] + k * (cov_jp / obs_var_ens) * a
    
    return new_anom

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LETKF-based analysis update with proper spread preservation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observations (even-indexed variables)
    - observed_indices: array of indices that are observed
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Block localization radius
    radius = 4
    
    updated_ensemble = ensemble.copy()
    
    # Process each observation sequentially with local block
    for obs_k in range(n_obs):
        obs_var_idx = obs_idx[obs_k]
        obs_val = observations[obs_k]
        obs_var = observation_variance
        
        # Local block: variables within radius of obs_var_idx
        local_indices = []
        for d in range(-radius, radius + 1):
            idx = (obs_var_idx + d) % n_vars
            local_indices.append(idx)
        
        # Remove duplicates preserving order
        seen = set()
        local_indices_unique = []
        for idx in local_indices:
            if idx not in seen:
                seen.add(idx)
                local_indices_unique.append(idx)
        
        # Find the position of obs_var_idx in the local block
        obs_pos = local_indices_unique.index(obs_var_idx)
        
        # Extract local ensemble
        local_ens = updated_ensemble[:, local_indices_unique].copy()
        
        # Compute local mean and anomalies
        local_mean = np.mean(local_ens, axis=0)
        local_anom = local_ens - local_mean
        
        # Observed variable anomaly
        obs_anom = local_anom[:, obs_pos]
        obs_var_ens = np.var(obs_anom)
        
        if obs_var_ens < 1e-12:
            continue
        
        # Kalman gain
        k = obs_var_ens / (obs_var_ens + obs_var)
        
        # Innovation
        innovation = obs_val - local_mean[obs_pos]
        
        # Update the mean
        local_mean[obs_pos] += k * innovation
        
        # Compute the LETKF-transformed anomalies
        new_anom = _letkf_transform_single_obs(local_anom, obs_pos, obs_var_ens, k)
        
        # Reconstruct the local ensemble
        local_ens_updated = local_mean + new_anom
        
        # Update the full ensemble
        updated_ensemble[:, local_indices_unique] = local_ens_updated
    
    # Ensure finiteness and valid range
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
