import numpy as np
import time

def _lorenz96_rhs(x, F):
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5*dt*k1, F)
    k3 = _lorenz96_rhs(x + 0.5*dt*k2, F)
    k4 = _lorenz96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    t_start = time.time()
    deadline = t_start + 100.0  # 20s safety margin before 120s limit
    
    n_members = ensemble.shape[0]
    n_vars = ensemble.shape[1]
    n_obs = len(observed_indices)
    
    # Convert to centered anomalies
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean  # (n_members, n_vars)
    
    # Compute ensemble covariance
    # Use (n-1) normalization
    C = (X.T @ X) / (n_members - 1)  # (n_vars, n_vars)
    
    # Localization: Gaussian kernel based on ring distance
    loc_width = 4.0  # localization width in grid points
    loc_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            dist = min(abs(i - j), n_vars - abs(i - j))
            loc_matrix[i, j] = np.exp(-0.5 * (dist / loc_width) ** 2)
    
    # Localized covariance
    C_loc = C * loc_matrix
    
    # Observation matrix
    H = np.zeros((n_obs, n_vars))
    for k, idx in enumerate(observed_indices):
        H[k, idx] = 1.0
    
    # Observation innovation
    y_obs = observations - H @ x_mean  # (n_obs,)
    
    # Observation covariance (diagonal)
    R = np.diag(np.full(n_obs, observation_variance))
    
    # Localized observation covariance to match localization
    R_loc = np.diag(np.full(n_obs, observation_variance))
    
    # Compute H C_loc H^T
    HC = H @ C_loc  # (n_obs, n_vars)
    S = HC @ H.T + R_loc  # (n_obs, n_obs)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain
    K = C_loc @ H.T @ S_inv  # (n_vars, n_obs)
    
    # Adaptive inflation: estimate from ensemble spread
    # Compute innovation variance
    innovation = y_obs  # (n_obs,)
    innov_var = np.var(innovation)
    expected_innov_var = np.mean(np.diag(S))
    
    # Inflation factor: if spread too low, inflate
    if expected_innov_var > 0:
        ratio = innov_var / expected_innov_var
        # Gentle inflation to maintain spread
        inflation = 1.0 + 0.1 * max(0, 1.0 - ratio)
    else:
        inflation = 1.0
    
    # Apply inflation to anomalies
    X_inflated = X * inflation
    
    # Recompute with inflated anomalies
    C_inflated = (X_inflated.T @ X_inflated) / (n_members - 1)
    C_loc_inflated = C_inflated * loc_matrix
    
    HC_inflated = H @ C_loc_inflated
    S_inflated = HC_inflated @ H.T + R_loc
    
    try:
        S_inv_inflated = np.linalg.inv(S_inflated)
    except np.linalg.LinAlgError:
        S_inv_inflated = np.linalg.pinv(S_inflated)
    
    K_inflated = C_loc_inflated @ H.T @ S_inv_inflated
    
    # Update mean
    x_new_mean = x_mean + K_inflated @ innovation
    
    # Update ensemble members
    # Use ensemble Kalman update: x_new_i = x_i + K (y_obs - H x_i)
    result = np.empty_like(ensemble)
    for m in range(n_members):
        x_m = ensemble[m]
        innov_m = observations - H @ x_m
        result[m] = x_m + K_inflated @ innov_m
    
    # Clamp to finite and within bounds
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    # Optional: second pass with refined parameters if time allows
    elapsed = time.time() - t_start
    if elapsed < 50.0:
        # Try a slightly different localization width
        for loc_w in [3.0, 5.0]:
            if time.time() > deadline:
                break
            loc_matrix2 = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    dist = min(abs(i - j), n_vars - abs(i - j))
                    loc_matrix2[i, j] = np.exp(-0.5 * (dist / loc_w) ** 2)
            
            C_loc2 = C * loc_matrix2
            HC2 = H @ C_loc2
            S2 = HC2 @ H.T + R_loc
            
            try:
                S_inv2 = np.linalg.inv(S2)
            except np.linalg.LinAlgError:
                S_inv2 = np.linalg.pinv(S2)
            
            K2 = C_loc2 @ H.T @ S_inv2
            
            result2 = np.empty_like(ensemble)
            for m in range(n_members):
                x_m = ensemble[m]
                innov_m = observations - H @ x_m
                result2[m] = x_m + K2 @ innov_m
            
            result2 = np.clip(result2, -100.0, 100.0)
            result2 = np.nan_to_num(result2, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Pick the one with smaller innovation norm
            innov1 = observations - np.mean(result, axis=0)[observed_indices]
            innov2 = observations - np.mean(result2, axis=0)[observed_indices]
            if np.sum(innov2**2) < np.sum(innov1**2):
                result = result2
    
    return result
