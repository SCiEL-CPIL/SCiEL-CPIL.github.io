import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Global Ensemble Kalman Filter with Gaspari-Cohn Localization and Adaptive Inflation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    x = ensemble.copy()
    
    # 1. Compute Mean and Anomalies
    x_mean = np.mean(x, axis=0)
    anomalies = x - x_mean
    
    # 2. Adaptive Inflation
    # Estimate inflation factor based on innovation variance
    S = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Model innovation variance (using raw spread for robustness)
    S_HH_raw = S[np.ix_(observed_indices, observed_indices)]
    var_inn_model = np.trace(S_HH_raw) / n_obs + observation_variance
    innovation = observations - x_mean[observed_indices]
    var_inn_obs = np.var(innovation, ddof=1) if n_obs > 1 else np.var(innovation)
    
    if var_inn_model > 0:
        inflation_factor = np.sqrt(var_inn_obs / var_inn_model)
    else:
        inflation_factor = 1.0
    
    # Clip inflation
    inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
    
    # Apply inflation
    x_inflated = x_mean + anomalies * inflation_factor
    anomalies_inflated = x_inflated - x_mean
    S_inflated = (anomalies_inflated.T @ anomalies_inflated) / (n_members - 1)
    
    # 3. Localization (Gaspari-Cohn)
    L_loc = 4.0
    loc_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            d = abs(i - j)
            d = min(d, n_vars - d)
            if d < L_loc:
                q = d / L_loc
                # GC5 polynomial
                loc_val = (
                    -1/2 * q**5 + 5/4 * q**4 - 5/12 * q**3 
                    + 1/4 * q**2 + 1
                )
                loc_matrix[i, j] = loc_val
                
    S_loc = S_inflated * loc_matrix
    
    # 4. Compute Kalman Gain
    S_HH_loc = S_loc[np.ix_(observed_indices, observed_indices)]
    S_xH_loc = S_loc[:, observed_indices]
    
    R = observation_variance * np.eye(n_obs)
    S_HH_total = S_HH_loc + R
    
    # Solve K * S_HH_total = S_xH_loc
    try:
        K = np.linalg.solve(S_HH_total.T, S_xH_loc.T).T
    except np.linalg.LinAlgError:
        K = S_xH_loc @ np.linalg.pinv(S_HH_total)
        
    # 5. Update
    innovation_global = observations - x_mean[observed_indices]
    x_mean_new = x_mean + K @ innovation_global
    
    anomalies_obs = anomalies_inflated[:, observed_indices]
    
    x_updated = np.zeros_like(x_inflated)
    for k in range(n_members):
        x_updated[k] = x_mean_new + anomalies_inflated[k] - K @ anomalies_obs[k]
        
    x_updated = np.clip(x_updated, -100.0, 100.0)
    x_updated = np.where(np.isfinite(x_updated), x_updated, 0.0)
    
    return x_updated
