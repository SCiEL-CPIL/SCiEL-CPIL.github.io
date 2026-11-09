"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed variables
    observed_indices : array-like
        Indices of observed variables (even indices: 0,2,4,...,18)
    observation_variance : float
        Variance of observation noise
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    ens = ensemble.copy().astype(np.float64)
    n_ens, n_var = ens.shape
    
    # Get observed indices as array
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Observation noise standard deviation
    obs_std = np.sqrt(observation_variance)
    
    # --- Localization: Gaussian with sigma = 2.8 ---
    loc_sigma = 2.8
    
    # --- Adaptive inflation ---
    # Initialize inflation factor (will be adjusted)
    inflation = 1.05
    
    # Compute ensemble mean and anomalies
    ens_mean = ens.mean(axis=0)
    anomalies = ens - ens_mean
    
    # Apply inflation to anomalies
    anomalies = inflation * anomalies
    ens = ens_mean + anomalies
    
    # --- Sequential update over observed variables ---
    for i in range(n_obs):
        if time.time() > deadline:
            break
        
        j = obs_idx[i]
        obs_val = observations[i]
        
        # Compute localization weights for variable j
        # Distance in periodic space (n_var = 20)
        dists = np.arange(n_var)
        dists = np.minimum(dists, n_var - dists)  # periodic distance
        loc_weights = np.exp(-0.5 * (dists / loc_sigma) ** 2)
        
        # Correlation between ensemble and observation at j
        # Use the ensemble values at j to compute correlation with all variables
        ens_j = ens[:, j]
        ens_j_mean = ens_j.mean()
        ens_j_anom = ens_j - ens_j_mean
        
        # For each variable k, compute correlation with j using localization
        # Simplified: use the ensemble covariance structure
        # C_kj = <anom_k * anom_j> / (std_k * std_j)
        
        # Compute ensemble standard deviations
        stds = ens.std(axis=0)
        stds[stds < 1e-10] = 1e-10
        std_j = stds[j]
        
        # Correlations with localization
        corr = np.zeros(n_var)
        for k in range(n_var):
            if k == j:
                corr[k] = 1.0
            else:
                # Compute correlation
                cov_kj = np.mean(ens[:, k] * ens_j, axis=0) - ens_mean[k] * ens_j_mean
                corr[k] = cov_kj / (stds[k] * std_j)
        
        # Apply localization
        corr *= loc_weights
        corr[j] = 1.0  # Ensure self-correlation is 1
        
        # --- Square root filter update ---
        # Update each ensemble member
        # Using the formula: x_a = x_f + C * P^{-1} * (y - H*x_f)
        # For scalar observation, this simplifies
        
        # Innovation
        innovation = obs_val - ens[:, j]
        
        # Innovation variance
        innov_var = np.var(innovation) + obs_std ** 2
        innov_var = max(innov_var, 1e-10)
        
        # Gain
        gain = np.var(ens[:, j]) / innov_var
        
        # Update all variables using correlations
        for k in range(n_var):
            if time.time() > deadline:
                break
            # Update variable k based on correlation with j
            update = gain * corr[k] * innovation
            ens[:, k] += update
        
        # --- Adaptive inflation adjustment ---
        # Compute RMSE at observed point
        rmse_obs = np.sqrt(np.mean((ens[:, j] - obs_val) ** 2))
        
        # Target RMSE should be close to observation std
        target_rmse = obs_std
        if rmse_obs > 1.5 * target_rmse:
            # Increase inflation if under-dispersive
            inflation = min(inflation * 1.02, 2.0)
        elif rmse_obs < 0.5 * target_rmse:
            # Decrease inflation if over-dispersive
            inflation = max(inflation * 0.98, 1.01)
    
    # Final inflation pass
    ens_mean_final = ens.mean(axis=0)
    ens = ens_mean_final + inflation * (ens - ens_mean_final)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    return ens
# EVOLVE-BLOCK-END
