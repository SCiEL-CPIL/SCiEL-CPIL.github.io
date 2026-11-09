"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Estimate background error variance (diagonal approximation for stability)
    # Add a small floor to prevent division by zero or extreme gains if ensemble collapses
    bg_var = np.var(ensemble[:, obs_idx], axis=0)
    bg_var = np.maximum(bg_var, 1e-6)
    
    # 2. Calculate Observation Variance
    # The task states N(0, 0.49) noise. 0.49 is likely the variance.
    # If 0.49 was std, variance would be 0.2401. Given "0.49" is close to 0.5, 
    # and typical Lo96 obs noise is around 0.1-0.5 std, let's assume 0.49 is variance.
    # However, often in these tasks, the parameter passed is the variance.
    # Let's trust the argument 'observation_variance'.
    obs_var = np.full(n_obs, observation_variance)
    
    # 3. Localization
    # Gaussian localization to handle the ring topology and local correlations.
    # Radius of influence ~ 2-3 grid points.
    r_loc = 2.5
    loc_weights = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            # Distance on the ring (periodic indices 0..19, obs are even: 0,2,4..)
            # Indices in obs_idx are actual state indices (0, 2, ..., 18)
            dist = abs(obs_idx[i] - obs_idx[j])
            dist = min(dist, n_vars - dist)
            # Gaussian decay
            loc_weights[i, j] = np.exp(-0.5 * (dist / r_loc) ** 2)
    
    # 4. Compute Ensemble Kalman Gain (EKG)
    # H is identity for observed indices.
    # P_bg = X_bg X_bg^T / (M-1)
    # S = H P_bg H^T + R
    # K = P_bg H^T S^-1
    # We want updated X = X_bg + K (y - H X_bg)
    
    # Background ensemble centered at mean
    bg_mean = np.mean(ensemble, axis=0)
    X_bg = ensemble - bg_mean  # (M, N)
    
    # H X_bg -> just pick observed columns
    X_bg_obs = X_bg[:, obs_idx]  # (M, N_obs)
    
    # Compute S = H P H^T + R
    # P_obs = (1/(M-1)) * X_bg_obs @ X_bg_obs.T
    # This is MxM. M=16, N_obs=10. Efficient.
    S = (X_bg_obs @ X_bg_obs.T) / (n_members - 1) + np.diag(obs_var)
    
    # Apply localization to S (diagonal or full matrix localization?)
    # Full matrix localization is more rigorous but can be tricky.
    # Standard practice: Localize the covariance P, or localize the gain K.
    # Let's localize the background error covariance contribution in S.
    # S_loc[i,j] = S[i,j] * loc_weights[i,j]
    # But we must ensure S remains positive definite.
    # Alternative: Localize the Gain K directly after calculation.
    
    # Calculate K without localization first
    # K = P H^T S^-1
    # P H^T = (1/(M-1)) * X_bg @ X_bg_obs.T  (M x N_obs)
    P_Ht = (X_bg @ X_bg_obs.T) / (n_members - 1)
    
    # Solve S^-1 P_Ht^T? No, K = P_Ht S^-1
    # We need to solve S K^T = P_Ht^T => K^T = S^-1 P_Ht^T
    try:
        K_T = np.linalg.solve(S, P_Ht.T)  # (N_obs, M)
        K = K_T.T  # (M, N_obs)
    except np.linalg.LinAlgError:
        # Fallback to naive nudge if singular
        result = ensemble.copy()
        result[:, obs_idx] += 0.6 * (observations - result[:, obs_idx])
        return result

    # Apply localization to the Gain K
    # K[m, i] is the gain for member m at obs i.
    # We want to dampen the influence of distant obs i on member m's update for *all* variables?
    # Actually, K is used to update ALL variables: X_new = X + K (y - H X_bg)
    # The localization should affect how much obs j influences the update of state variable i.
    # In EKG, K is (M, N_obs). The update for variable v is sum_j K[:, j] * (y_j - H_j X_bg[:, j])
    # Wait, the standard EKG update is:
    # x_a = x_b + K (y - H x_b)
    # Here x_a and x_b are vectors of length N_vars.
    # K is (N_vars, N_obs).
    # My current K is (M, N_obs). This is the "Ensemble Kalman Gain" form where we update each member directly.
    # Formula: x_a^m = x_b^m + K^m (y - H x_b^m)
    # Where K^m is a row vector of length N_obs.
    # This is correct.
    
    # How to localize?
    # The gain K[m, j] determines how much obs j corrects member m.
    # The correction is then applied to ALL variables of member m.
    # This implies that obs j affects all variables.
    # To localize spatially, we need the gain to depend on the target variable index too.
    # The standard "Local" Ensemble Square Root Filter (LESRF) or Local EnKF does this by
    # updating each variable independently using only nearby observations.
    
    # Let's switch to a variable-by-variable update strategy (Local EnKF).
    # For each variable v in 0..19:
    #   1. Identify nearby observations j where loc_weight(v, j) > threshold.
    #   2. Compute local gain for v using only those obs.
    #   3. Update ensemble members for variable v.
    
    # This is computationally heavier (20 iterations) but M=16 is small.
    # It is also more robust to the "chaotic" nature where distant obs are irrelevant.
    
    result = ensemble.copy()
    bg_mean_all = np.mean(ensemble, axis=0)
    
    # Pre-compute distances between all state vars and obs vars
    # obs_idx are [0, 2, 4, ..., 18]
    
    for v in range(n_vars):
        # Find relevant obs
        # Distance from state var v to each obs
        dists = np.array([min(abs(v - oj), n_vars - abs(v - oj)) for oj in obs_idx])
        weights = np.exp(-0.5 * (dists / r_loc) ** 2)
        
        # Only use obs with significant weight
        mask = weights > 0.1
        if not np.any(mask):
            continue
            
        obs_idx_local = obs_idx[mask]
        weights_local = weights[mask]
        n_loc = len(obs_idx_local)
        
        # Background ensemble for this variable and local obs
        X_v = ensemble[:, v]  # (M,)
        X_obs = ensemble[:, obs_idx_local]  # (M, n_loc)
        
        # Center
        X_v_c = X_v - bg_mean_all[v]
        X_obs_c = X_obs - bg_mean_all[obs_idx_local]
        
        # Local Covariances
        # P_vv = var(X_v)
        # P_vo = cov(X_v, X_obs)
        # P_oo = cov(X_obs, X_obs)
        
        # We can compute the gain directly
        # S = P_oo + R_local
        # R_local = diag(obs_var) but scaled by localization? 
        # Localization is usually applied to P_oo.
        # Let's apply localization to the covariance P_oo.
        
        P_oo = (X_obs_c @ X_obs_c.T) / (n_members - 1)
        # Apply localization to P_oo
        P_oo_loc = P_oo * weights_local[:, None] * weights_local[None, :]
        
        # Ensure PD
        P_oo_loc += np.eye(n_loc) * 1e-6
        
        S = P_oo_loc + np.diag(obs_var[mask])
        
        P_vo = (X_v_c @ X_obs_c.T) / (n_members - 1)
        
        # Gain K_v = P_vo S^-1
        try:
            K_v = np.linalg.solve(S.T, P_vo.T).T  # (1, n_loc)
        except np.linalg.LinAlgError:
            continue
            
        K_v = K_v[0]  # (n_loc,)
        
        # Innovation
        innovation = observations[mask] - bg_mean_all[obs_idx_local]
        
        # Update
        # x_a = x_b + K_v * innovation
        # But we must also account for the fact that the ensemble spread is being reduced.
        # The standard EKG update:
        # x_a^m = x_b^m + K_v (y - H x_b^m)
        # Note: H x_b^m is the observation from member m.
        # innovation_m = y - X_obs_c[:, m] - bg_mean_all[obs_idx_local] ?
        # No, innovation is y - H x_b.
        # Let's use the member-specific innovation for better consistency.
        
        # x_b_obs_m = ensemble[:, obs_idx_local] (already have X_obs)
        # innov_m = observations[mask] - X_obs[m, :]
        
        # Update each member
        for m in range(n_members):
            innov_m = observations[mask] - X_obs[m, :]
            result[m, v] = ensemble[m, v] + np.dot(K_v, innov_m)
            
        # Inflation
        # Check if the ensemble variance at this variable has collapsed
        # We need to estimate the "true" variance or use a relative inflation.
        # Simple adaptive inflation:
        # If the ratio of background variance to innovation variance is too high, inflate.
        # Or simpler: always apply a small inflation factor to maintain spread.
        # Given the chaotic system, a static inflation factor of 1.01-1.02 is often needed.
        # Let's apply a gentle inflation to the whole ensemble after all updates.
        
    # Global Inflation
    # Calculate current mean and variance
    new_mean = np.mean(result, axis=0)
    new_var = np.var(result, axis=0)
    
    # Estimate expected variance from background?
    # We don't have the truth.
    # Let's use a heuristic: if the ensemble is too tight, inflate.
    # A common method is to compare the ensemble spread to the observation error.
    # But a simpler, robust approach for Lo96 is constant inflation.
    # Let's try a mild constant inflation.
    inflation_factor = 1.01
    result = new_mean + inflation_factor * (result - new_mean)
    
    # Safety check: finite and bounds
    if not np.all(np.isfinite(result)):
        return ensemble.copy()
        
    result = np.clip(result, -100, 100)
    
    # Time check
    if time.time() > deadline:
         pass # We are done, return
        
    return result

# EVOLVE-BLOCK-END
