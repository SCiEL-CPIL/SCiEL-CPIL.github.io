"""Lorenz-96 ensemble assimilation with Local Ensemble Transform Kalman Filter (LETKF)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LETKF-based ensemble analysis update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    R = observation_variance if np.isscalar(observation_variance) else np.mean(observation_variance)
    
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Define local block size for LETKF
    # A block of 6 variables covers 3 observed variables (even indices)
    # With 20 variables, we can have blocks centered around each observed variable
    # Or we can just process the whole state if it's small enough, but localization is better.
    # Let's use a block width of 6.
    block_width = 6
    half_width = block_width // 2
    
    # Pre-compute localization weights for the block
    # We use a Gaussian decay with a correlation length of 3 (half the block width)
    corr_len = 3.0
    dists = np.arange(-half_width, half_width + 1)
    loc_weights = np.exp(-0.5 * (dists / corr_len) ** 2)
    
    # Adaptive inflation parameters
    # We want the innovation standard deviation to be close to sqrt(R)
    # If it's larger, the ensemble is too tight -> inflate
    # If it's smaller, the ensemble is too wide -> deflate
    inflation_scale = 1.0
    
    # Process local blocks
    # We iterate over the observed indices and apply the LETKF update to a local window
    # To avoid double-counting or interference, we can simply process each observed variable's local window.
    # Since the windows overlap, we need to be careful. 
    # A standard approach is to process disjoint blocks or to use the full state but with localized covariances.
    # Given N=20 is small, we can actually afford to do a full state LETKF with localization weights applied to the covariance.
    # However, the "Local" in LETKF usually implies processing subsets of the state independently.
    # Let's process each observed variable with a local window of size 6.
    # We will update the ensemble in-place.
    
    for j in range(n_obs):
        if time.time() > deadline:
            break
            
        obs_j = obs[j]
        obs_pos = obs_idx[j]
        
        # Define the local window indices
        # Indices relative to obs_pos
        rel_indices = np.arange(-half_width, half_width + 1)
        abs_indices = (obs_pos + rel_indices) % n_vars
        
        # Extract local ensemble anomalies
        # X_local shape: (n_members, block_width)
        X_local = ens[:, abs_indices]
        X_mean = np.mean(X_local, axis=0)
        X_anom = X_local - X_mean
        
        # Observation operator H_local
        # H maps the local state to the single observation obs_j
        # The observation is at index 0 in the local window (since rel_indices starts at -half_width, 
        # and obs_pos is at index half_width in abs_indices? No, rel_indices is -3..3, obs_pos is 0 relative to itself.
        # So the observation is at index half_width in the local array.
        H_idx = half_width
        
        # Compute the local ensemble mean observation
        H_ens = X_anom[:, H_idx]
        
        # Innovation
        innovation = obs_j - (X_mean[H_idx])
        
        # Compute the ensemble variance of the observation
        # S = var(H_ens) + R
        var_obs = np.var(H_ens, ddof=1) if n_members > 1 else 0.0
        S = var_obs + R
        
        if S <= 0:
            continue
            
        # Compute the SVD of the local anomaly matrix
        # X_anom = U * S_diag * Vt
        # We only need the right singular vectors Vt (or V) for the transform
        # X_anom is (n_members, block_width). 
        # We want to transform the anomalies.
        # The transform matrix T is applied to the anomalies: X_new = X_mean + X_anom @ T
        # T = V @ diag(t) @ Vt, where t is the scaling factor.
        
        # For efficiency and stability, we use the economy SVD
        U, s, Vt = np.linalg.svd(X_anom, full_matrices=False)
        
        # The eigenvalues of the covariance matrix P = X_anom^T X_anom / (n-1) are s^2 / (n-1)
        # The transform scaling factor t_i is given by:
        # t_i = (1 + k_i * s_i^2)^(-1/2)
        # where k_i is related to the observation operator.
        # In the LETKF formulation for a single observation:
        # The gain is applied in the subspace.
        # Let's use the standard LETKF formula for a single observation.
        # The transform T is constructed from the eigenvectors of the covariance.
        
        # Actually, a simpler and robust implementation for a single observation in a local block:
        # We can use the "additive" form or the "transform" form.
        # The transform form is:
        # T = I - (1 - 1/sqrt(1 + H^T P^{-1} H)) * (P H H^T P) / (H^T P H + R)
        # This is complex to compute with SVD directly for a single obs.
        
        # Alternative: Use the sequential square-root update but restricted to the local block.
        # This is computationally cheap and effective.
        
        # Let's stick to the sequential LESRF but restricted to the local block.
        # This avoids the complexity of the full LETKF transform construction for a single obs
        # while still benefiting from localization.
        
        # Compute cross-covariance between all local variables and the observed variable
        # P_local = X_anom^T X_anom / (n-1)
        # Cross-cov P_x_obs = X_anom^T H_ens / (n-1)
        
        cross_cov = np.dot(X_anom.T, H_ens) / (n_members - 1.0)
        
        # Apply localization to the cross-covariance
        # The localization weights are applied to the distance from the observed variable
        # The observed variable is at index H_idx in the local array.
        # The distances are |i - H_idx| for i in 0..block_width-1
        local_dists = np.abs(np.arange(block_width) - H_idx)
        local_weights = np.exp(-0.5 * (local_dists / corr_len) ** 2)
        
        cross_cov_loc = cross_cov * local_weights
        
        # Update the local ensemble
        # X_new = X_old + P_x_obs * K * (obs - H_ens)
        # K = P_obs_obs / S = var_obs / S
        K = var_obs / S
        
        # Update anomalies
        # X_anom_new = X_anom + cross_cov_loc * K * (obs_j - (X_mean[H_idx] + H_ens))
        # Wait, the innovation for each member is (obs_j - (X_mean[H_idx] + H_ens[m]))
        # So X_anom_new[:, i] = X_anom[:, i] + cross_cov_loc[i] * K * (obs_j - X_mean[H_idx] - H_ens)
        
        # Let's vectorize this update
        # innovation_vec = obs_j - X_mean[H_idx] - H_ens  # shape (n_members,)
        innovation_vec = obs_j - X_mean[H_idx] - H_ens
        
        # Update X_anom
        # X_anom_new = X_anom + np.outer(innovation_vec, cross_cov_loc * K)
        X_anom_new = X_anom + np.outer(innovation_vec, cross_cov_loc * K)
        
        # Update the local ensemble
        X_local_new = X_mean + X_anom_new
        
        # Assign back to the global ensemble
        ens[:, abs_indices] = X_local_new
        
    # Final adaptive inflation
    # Compute the global innovation statistics
    # We can estimate the innovation variance from the last few updates or just use a fixed small inflation
    # To be safe, apply a small multiplicative inflation to prevent collapse
    ens_mean = np.mean(ens, axis=0)
    ens_centered = ens - ens_mean
    
    # Simple adaptive inflation based on ensemble spread
    # If the spread is very small, inflate more
    # For now, use a conservative fixed inflation
    inflation = 1.05
    ens = ens_mean + inflation * ens_centered
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finite values
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    # Check time
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
    
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
