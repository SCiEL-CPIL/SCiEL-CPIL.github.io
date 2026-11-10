"""Local ESRF with adaptive inflation for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _local_esrf_update(ens, obs_idx, obs_val, obs_var, window_size=2):
    """
    Perform a local ESRF update for a single observed variable.
    Updates the local window around obs_idx.
    """
    M, N = ens.shape
    i = obs_idx
    
    # Local indices with periodic boundary conditions
    # We use a window of +/- window_size
    local_indices = [(i - window_size + j) % N for j in range(2 * window_size + 1)]
    
    # Extract local ensemble (M x (2*window_size+1))
    local_ens = ens[:, local_indices]
    
    # The observed variable is at the center of the window
    # Index in local_ens corresponding to 'i'
    center_idx = window_size
    
    # Compute local mean and anomalies
    local_mean = local_ens.mean(axis=0)
    local_anom = local_ens - local_mean
    
    # Local covariance (2*window_size+1 x 2*window_size+1)
    local_cov = (local_anom.T @ local_anom) / (M - 1)
    
    # Observation variance
    R = obs_var
    
    # Innovation covariance S = local_cov[center, center] + R
    S = local_cov[center_idx, center_idx] + R
    
    # Gain vector K (2*window_size+1 x 1)
    # K = local_cov[:, center_idx] / S
    K = local_cov[:, center_idx] / S
    
    # Innovation for each ensemble member
    # obs_val is scalar, local_ens[:, center_idx] is (M,)
    innovations = obs_val - local_ens[:, center_idx]
    
    # Update local ensemble
    # local_ens_new = local_ens + innovations @ K.T
    # K.T is (1 x L), innovations is (M x 1), result is (M x L)
    local_ens_updated = local_ens + np.outer(innovations, K)
    
    # Write back to full ensemble
    ens[:, local_indices] = local_ens_updated
    
    return ens


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ESRF with adaptive inflation for Lorenz-96.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # 1. Adaptive Inflation
    # Calculate spread at observed locations
    obs_std = np.sqrt(observation_variance)
    ens_spread = np.std(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    
    # Ratio of ensemble spread to observation error
    ratio = mean_spread / obs_std if obs_std > 0 else 1.0
    
    # Inflation factor
    # If ratio is low, inflate more. If high, keep close to 1.
    # Target ratio ~ 1.0 - 1.5
    if ratio < 1.0:
        inflation = 1.0 + 0.2 * (1.0 - ratio)
    else:
        inflation = 1.0 + 0.05 * np.exp(-(ratio - 1.0)) # Gentle decay for over-dispersed
    
    inflation = np.clip(inflation, 1.0, 1.5)
    
    # Apply multiplicative inflation
    mean = ensemble.mean(axis=0)
    inflated = mean + inflation * (ensemble - mean)
    
    # 2. Local ESRF Updates
    # We update sequentially for each observed index.
    # Order can matter, but for local updates, it's usually fine.
    # We can also randomize order or do them in parallel if independent, 
    # but they overlap (window size 2 -> overlap of 4). 
    # Sequential is safer for stability in small ensembles.
    
    updated = inflated.copy()
    
    # To improve efficiency and stability, we can process updates.
    # Since N=20 and K=10, this is very fast.
    for k in range(K):
        if time.time() > deadline:
            break
            
        idx = observed_indices[k]
        val = observations[k]
        var = observation_variance
        
        # Perform local update
        updated = _local_esrf_update(updated, idx, val, var, window_size=2)
        
    # 3. Safety Checks
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
