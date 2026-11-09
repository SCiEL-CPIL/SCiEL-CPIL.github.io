"""Ensemble Kalman Filter with observation perturbation for Lorenz-96 assimilation."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_factor = 1.0
_inflation_min = 1.0
_inflation_max = 2.0
_inflation_alpha = 0.01  # slow adaptation
_prev_analysis_cov_diag = None

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for a single state vector."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _lorenz96_rhs_vec(X, F):
    """Compute Lorenz-96 RHS for an ensemble (each row is a state)."""
    n = X.shape[1]
    # Periodic indices
    x_im2 = np.roll(X, 2, axis=1)
    x_im1 = np.roll(X, 1, axis=1)
    x_ip1 = np.roll(X, -1, axis=1)
    return (x_ip1 - x_im2) * x_im1 - X + F

def _rk4_step(X, F, dt):
    """RK4 step for ensemble."""
    k1 = _lorenz96_rhs_vec(X, F)
    k2 = _lorenz96_rhs_vec(X + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs_vec(X + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs_vec(X + dt * k3, F)
    return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update with observation perturbation.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of states
    observations : (10,) array - noisy observations of even-indexed coordinates
    observed_indices : array-like - indices of observed variables (even indices)
    observation_variance : float - variance of observation noise
    
    Returns:
    --------
    updated ensemble with shape (16, 20)
    """
    global _inflation_factor, _prev_analysis_cov_diag
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    N = ensemble.shape[0]  # ensemble size (16)
    n = ensemble.shape[1]  # state dimension (20)
    m = len(observed_indices)  # number of observations (10)
    
    # --- Step 1: Multiplicative inflation (adaptive) ---
    # Inflate the ensemble anomalies to account for model error
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    ensemble_inflated = mean + _inflation_factor * anomalies
    
    # --- Step 2: Observation perturbation ---
    # Add random perturbations to observations to preserve ensemble spread
    # Use a fixed seed derived from current state for reproducibility within a run
    # But we want different perturbations each call, so use numpy's global RNG
    rng = np.random.default_rng()
    obs_perturbed = observations + rng.standard_normal((N, m)) * np.sqrt(observation_variance)
    
    # --- Step 3: Compute analysis using linear regression (EnKF) ---
    # For each ensemble member, we regress the observed variables onto the full state
    # The analysis is: x_a = x_f + K (y_obs - H x_f) where K is the Kalman gain
    
    # Let's use the standard EnKF formulation:
    # X_a = X_f + (Y_pert - H X_f) (H X_f - H X_f)^+ (I - H X_f)^+ ... 
    # Actually, the standard EnKF with observation perturbation:
    # For each member i:
    #   x_a_i = x_f_i + sum_j K_j (y_pert_i,j - H x_f_i,j)
    # where K is the regression coefficient
    
    # More precisely, using the ensemble regression approach:
    # Let H be the observation operator (selects observed_indices)
    # X_obs = ensemble_inflated[:, observed_indices]  # (N, m)
    # Y_pert = obs_perturbed  # (N, m)
    # 
    # The analysis for unobserved variables:
    # x_a = x_f + C_xy * C_yy^{-1} * (y_pert - y_f)
    # where C_xy is the cross-covariance between full state and observations
    # and C_yy is the observation covariance
    
    # Using ensemble regression:
    X_full = ensemble_inflated  # (N, n)
    X_obs = X_full[:, observed_indices]  # (N, m)
    
    # Center the data
    X_full_c = X_full - X_full.mean(axis=0)  # (N, n)
    X_obs_c = X_obs - X_obs.mean(axis=0)  # (N, m)
    Y_c = Y_pert = obs_perturbed - obs_perturbed.mean(axis=0)  # (N, m)
    
    # Compute regression coefficients: K = C_xy * C_yy^{-1}
    # C_xy = (1/(N-1)) X_full_c^T X_obs_c  # (n, m)
    # C_yy = (1/(N-1)) X_obs_c^T X_obs_c  # (m, m)
    # K = C_xy * C_yy^{-1}  # (n, m)
    
    # To avoid explicit inversion, solve: X_obs_c K^T = X_full_c^T
    # i.e., K^T = (X_obs_c^T X_obs_c)^{-1} X_obs_c^T X_full_c
    # K = X_full_c^T (X_obs_c^T X_obs_c)^{-1} X_obs_c^T ... no wait
    
    # Let's do it directly:
    # We want K such that for each unobserved variable j:
    #   delta_x_j = sum_i K[j,i] * delta_y_i
    # where delta_y = y_pert - y_f
    
    # Using least squares: X_full_c = X_obs_c * K^T + noise
    # So K^T = pinv(X_obs_c) * X_full_c
    # K = pinv(X_obs_c).T * X_full_c.T ... 
    
    # Actually: X_full_c (N,n) ≈ X_obs_c (N,m) * K^T (m,n)
    # So K^T = argmin ||X_full_c - X_obs_c * B||_F, where B = K^T
    # B = pinv(X_obs_c) @ X_full_c
    # K = B.T
    
    try:
        B = np.linalg.pinv(X_obs_c) @ X_full_c  # (m, n)
        K = B.T  # (n, m)
    except np.linalg.LinAlgError:
        # Fallback to simple nudging
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        return result
    
    # Compute innovation: delta_y = y_pert - y_f (for each member)
    delta_y = obs_perturbed - X_obs  # (N, m)
    
    # Analysis: x_a = x_f + delta_y @ K
    # Note: delta_y is (N, m), K is (n, m), so delta_y @ K.T is (N, n)
    X_analysis = X_full + delta_y @ K
    
    # Ensure observed variables are properly updated
    # The regression should already handle this, but let's verify
    # For observed variables, the analysis should be close to the perturbed observations
    
    # --- Step 4: Adaptive inflation update ---
    # Estimate the implied observation variance from the analysis
    X_analysis_obs = X_analysis[:, observed_indices]
    analysis_obs_var = np.var(X_analysis_obs, axis=0)  # (m,)
    mean_analysis_obs_var = np.mean(analysis_obs_var)
    
    # If the analysis observation variance is too low, increase inflation
    # If it's too high, decrease it
    target_var = observation_variance
    if mean_analysis_obs_var > 0:
        ratio = target_var / mean_analysis_obs_var
        # Slowly adjust inflation
        _inflation_factor *= (1.0 + _inflation_alpha * (ratio - 1.0))
        _inflation_factor = np.clip(_inflation_factor, _inflation_min, _inflation_max)
    
    # Check deadline
    if time.time() > deadline:
        # Return the analysis we have
        pass
    
    # Clip to valid range
    X_analysis = np.clip(X_analysis, -100.0, 100.0)
    
    # Ensure finite values
    X_analysis = np.nan_to_num(X_analysis, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_analysis
