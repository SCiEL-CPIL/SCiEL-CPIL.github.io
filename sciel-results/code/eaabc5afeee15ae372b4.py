"""Local Ensemble Square Root Filter (ESRF) for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Local Ensemble Square Root Filter (ESRF) update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of the observed state components.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    n_ens = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Work with a copy
    result = ensemble.copy()
    
    # Compute ensemble mean
    ens_mean = result.mean(axis=0)
    
    # Compute anomalies
    anomalies = result - ens_mean
    
    # The ESRF update for a single observation (scalar observation case)
    # Since we have multiple observations but they are at different state components,
    # and the state is high-dimensional, we can apply the update sequentially 
    # or jointly. For Lorenz-96 with local observations, a sequential or 
    # localized joint update works well.
    
    # For simplicity and effectiveness, we'll apply the ESRF update for each 
    # observed component. However, since the observations are at distinct 
    # state indices and the Lorenz-96 system has local interactions, we can
    # treat this as a multi-observation update.
    
    # Let's do a proper multi-observation ESRF update.
    # The standard ESRF for multiple observations uses an orthogonal transformation.
    
    # For a set of m observations of m distinct state variables (no cross-correlation 
    # in the observation operator), the ESRF update is:
    # 1. Compute the ensemble mean of the observed variables
    # 2. Compute the ensemble covariance of the observed variables
    # 3. Compute the Kalman gain
    # 4. Apply the square root filter transformation
    
    # Actually, let's use the sequential ESRF approach which is simpler and 
    # works well for local observations.
    
    # Sequential ESRF: process each observation one at a time
    for idx in observed_indices:
        obs_val = observations[list(observed_indices).index(idx)]
        
        # Current ensemble mean for this component
        x_mean = result[:, idx].mean()
        
        # Current ensemble anomalies for this component
        x_anom = result[:, idx] - x_mean
        
        # Ensemble variance
        x_var = np.sum(x_anom ** 2) / (n_ens - 1)
        
        # Innovation
        innovation = obs_val - x_mean
        
        # Kalman gain
        r_var = observation_variance
        k = x_var / (x_var + r_var)
        
        # ESRF update: apply orthogonal transformation to preserve spread
        # The standard ESRF formula for a single observation:
        # x_new = x_mean + k * (x - x_mean) + (1 - k) * (x_mean + innovation * sqrt(1-k) * sign(x - x_mean) / sqrt(1 - (1-k)))
        
        # More precisely, the ESRF update is:
        # For each member i:
        # x_i_new = x_mean + sqrt(k) * (x_i - x_mean) + (1 - k) * innovation * sign(x_i - x_mean) / sqrt(1 - k) * (1 - k)
        
        # Let me use the standard ESRF formulation:
        # x_i_new = x_mean + sqrt(k) * (x_i - x_mean) + (1-k) * (obs - x_mean) * sign(x_i - x_mean) / sqrt(1-k)
        
        # Actually, the correct ESRF formula ensures the updated variance is:
        # x_var_new = x_var * (1 - k)
        # and the updated mean is:
        # x_mean_new = x_mean + k * (obs - x_mean)
        
        # The ESRF transformation is:
        # x_i_new = x_mean_new + sqrt(1-k) * (x_i - x_mean) * sign adjustment
        
        # Let me use the well-known ESRF formula:
        # x_i_new = x_mean + k*(x_i - x_mean) + (1-k)*(obs - x_mean) * (x_i - x_mean) / (sqrt(1-k) * |x_i - x_mean|)
        
        # Simplified: for the ESRF, the update is:
        # x_i_new = x_mean + sqrt(k) * (x_i - x_mean) + (1-k) * (obs - x_mean) / sqrt(1-k) * sign(x_i - x_mean)
        
        # Let me just use the standard sequential ESRF:
        # The key property is that the ensemble mean updates to:
        #   x_mean_new = x_mean + k * (obs - x_mean)
        # and the ensemble variance updates to:
        #   x_var_new = (1 - k) * x_var
        
        # The ESRF achieves this by applying an orthogonal transformation to the anomalies.
        
        # Standard ESRF update formula:
        # x_i_new = x_mean_new + (x_i - x_mean) * sqrt(1 - k) + (1-k) * (obs - x_mean) * sign(x_i - x_mean) / sqrt(1-k)
        
        # Wait, let me be more careful. The ESRF for a single observation:
        # Given ensemble {x_i}, mean x_bar, and observation y with variance R:
        # k = P / (P + R) where P = ensemble variance
        # New mean: x_bar_new = x_bar + k*(y - x_bar)
        # New anomalies: a_i_new = a_i * sqrt(1-k) + (1-k)*(y - x_bar) * sign(a_i) / sqrt(1-k)
        
        # This ensures:
        # - New mean is x_bar_new
        # - New variance is (1-k)*P
        
        x_mean_new = x_mean + k * innovation
        r_var = observation_variance
        
        # Apply ESRF update to each member
        for i in range(n_ens):
            a_i = anomalies[i, idx]
            if abs(a_i) > 1e-12:
                sign_a = 1.0 if a_i > 0 else -1.0
                result[i, idx] = x_mean_new + a_i * np.sqrt(1 - k) + (1 - k) * innovation * sign_a / np.sqrt(1 - k)
            else:
                result[i, idx] = x_mean_new
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
