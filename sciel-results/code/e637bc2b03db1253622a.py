import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Sequential Ensemble Kalman Filter (EnKF) with adaptive inflation
    for Lorenz-96 partial-observation assimilation.
    
    Uses local neighborhoods (cosine localization) to avoid spurious correlations.
    Processes observations sequentially, updating only affected state variables.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    obs_var = observation_variance
    x = ensemble.copy().astype(np.float64)
    
    # Adaptive inflation state (persistent across calls via module-level dict)
    if not hasattr(analyze, '_inflation'):
        analyze._inflation = 1.0
    
    # Apply previous inflation
    if analyze._inflation > 1.0:
        mean_prev = np.mean(x, axis=0)
        x = mean_prev + analyze._inflation * (x - mean_prev)
    
    # Process observations sequentially
    for obs_idx in observed_indices:
        # Define local neighborhood: variables within distance 3 of obs_idx
        # For L96 with 20 vars, use a window
        half_width = 3
        local_vars = []
        for d in range(-half_width, half_width + 1):
            idx = (obs_idx + d) % n_vars
            if idx not in local_vars:
                local_vars.append(idx)
        
        # Get background mean and deviations for local variables
        x_local = x[:, local_vars]
        mean_local = np.mean(x_local, axis=0)
        dev_local = x_local - mean_local  # (n_members, n_local)
        
        # Observation deviation
        y = observations[obs_idx]
        
        # Background observation (for the specific observed variable)
        obs_pos_in_local = local_vars.index(obs_idx)
        x_obs_bg = mean_local[obs_pos_in_local]
        
        # Innovation
        innovation = y - x_obs_bg
        
        # Background variance at observed variable
        var_obs = np.var(x_local[:, obs_pos_in_local], ddof=1)
        var_obs = max(var_obs, 1e-8)
        
        # Cross-covariance between local vars and observed var
        cross_cov = np.cov(dev_local[:, :], dev_local[:, obs_pos_in_local], rowvar=False, bias=False)
        # cross_cov is (n_local,) vector: cov(x_j, x_obs) for each j in local_vars
        # Actually np.cov with rowvar=False gives (n_local, n_local) matrix
        # Let's do it manually for clarity
        cross_cov = (dev_local.T @ dev_local[:, obs_pos_in_local]) / (n_members - 1)  # (n_local,)
        
        # Kalman gain
        S = var_obs + obs_var
        K = cross_cov / S  # (n_local,)
        
        # Update local variables
        # x_a_j = x_b_j + K_j * (y - x_b_obs)
        x[:, local_vars] = x_local + K * innovation
        
        # Re-normalize: ensure the observed variable is updated correctly
        # The above should already do this, but let's verify
        # x_obs_new = x_obs_bg + K[obs_pos_in_local] * innovation
        # K[obs_pos_in_local] = var_obs / (var_obs + obs_var)
        # So x_obs_new = x_obs_bg + var_obs/(var_obs+obs_var) * (y - x_obs_bg)
        # = (obs_var * x_obs_bg + var_obs * y) / (var_obs + obs_var)
        # This is the standard Kalman update for the observed variable
    
    # Adaptive inflation
    # Compute ensemble spread vs observation error
    mean_a = np.mean(x, axis=0)
    
    # For observed variables, compute the ensemble mean error
    obs_mean = mean_a[observed_indices]
    obs_err = observations - obs_mean
    obs_err_std = np.std(obs_err)
    
    # Compute ensemble spread at observed variables
    spread = np.std(x[:, observed_indices], axis=0, ddof=1)
    spread_mean = np.mean(spread)
    
    # Inflation factor: adjust to match spread to observed error
    # Target: spread should be comparable to sqrt(obs_var)
    # If spread is too small, inflate
    if spread_mean > 1e-8:
        ratio = obs_err_std / spread_mean
        # Clip ratio to avoid extreme inflation
        ratio = np.clip(ratio, 0.8, 2.0)
        # Smooth the inflation factor
        analyze._inflation = 0.7 * analyze._inflation + 0.3 * ratio
        analyze._inflation = np.clip(analyze._inflation, 0.95, 1.15)
    else:
        analyze._inflation = 1.0
    
    # Apply inflation
    if analyze._inflation > 1.001:
        x = mean_a + analyze._inflation * (x - mean_a)
    
    # Safety: ensure finite values and within bounds
    if not np.all(np.isfinite(x)):
        x = ensemble.copy()
        # Fallback: simple nudging
        x[:, observed_indices] += 0.5 * (observations - ensemble[:, observed_indices])
    
    # Clip to [-100, 100]
    x = np.clip(x, -100.0, 100.0)
    
    return x
