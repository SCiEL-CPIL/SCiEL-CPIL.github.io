import numpy as np
import time

def _lorenz96_step(x, F, dt=0.01, steps=5):
    """Propagate Lorenz-96 state by RK4 for `steps` sub-steps of `dt`."""
    n = len(x)
    for _ in range(steps):
        def deriv(x):
            return (np.roll(x, 1) - np.roll(x, 2)) * np.roll(x, -1) - x + F
        k1 = deriv(x)
        k2 = deriv(x + 0.5 * dt * k1)
        k3 = deriv(x + 0.5 * dt * k2)
        k4 = deriv(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed variables
    observed_indices: array of indices that are observed (even indices)
    observation_variance: scalar variance of observation noise
    
    Returns updated ensemble of shape (16, 20).
    """
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    N = ensemble.shape[1]  # 20 variables
    M = ensemble.shape[0]  # 16 members
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Estimate F from the ensemble mean dynamics (rough estimate)
    # Use the fact that in steady state, mean of x is roughly F/2 for L96
    # But we can estimate F more directly from the equation
    x_mean = np.mean(ensemble, axis=0)
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # At quasi-steady state, mean dx ~ 0, so F ~ mean(x_i - (x_{i+1}-x_{i-2})*x_{i-1})
    # Actually let's just use a reasonable default and adjust
    # For L96 with F=6,8,10, the mean state is roughly F/2
    F_est = np.mean(x_mean) * 2.0
    # Clamp to reasonable range
    F_est = np.clip(F_est, 4.0, 12.0)
    
    # --- Localization ---
    # Compute localization matrix for observation error covariance
    sigma_loc = 3.0  # correlation length
    
    # --- Adaptive Inflation ---
    # Compute ensemble variance at observed points
    obs_ens = ensemble[:, obs_idx]
    ens_var_obs = np.var(obs_ens, axis=0, ddof=1)
    obs_mean = np.mean(obs_ens, axis=0)
    
    # Innovation
    innovation = observations - obs_mean
    
    # Inflation factor: adaptively inflate if ensemble is too narrow
    # Compare observed innovation variance to expected
    innov_var = np.var(innovation, ddof=1) if n_obs > 1 else np.mean(innovation**2)
    expected_innov_var = np.mean(ens_var_obs) + observation_variance
    
    # Inflation factor (adaptive)
    if expected_innov_var > 1e-10:
        ratio = innov_var / expected_innov_var
        # Inflate if ratio > 1 (observations more spread than ensemble)
        inflation = 1.0 + 0.5 * max(0.0, ratio - 1.0)
        # Deflate slightly if ratio < 0.8 (ensemble too wide)
        if ratio < 0.8:
            inflation = 1.0 - 0.2 * (0.8 - ratio)
    else:
        inflation = 1.0
    
    inflation = np.clip(inflation, 0.8, 2.0)
    
    # Apply inflation to observed coordinates
    ens_mean_obs = np.mean(ensemble[:, obs_idx], axis=0)
    ensemble[:, obs_idx] = ens_mean_obs + inflation * (ensemble[:, obs_idx] - ens_mean_obs)
    
    # --- Deterministic Ensemble Square Root Filter (LESRF) ---
    # Use the ensemble to estimate the error covariance
    # K = P H^T (H P H^T + R)^-1  (Kalman gain)
    # For efficiency with 16 members and 10 obs, we can use the ensemble form
    
    # Center the ensemble
    ens_mean = np.mean(ensemble, axis=0)
    X = ensemble - ens_mean  # (M, N)
    
    # Observation operator
    # H selects observed indices
    H = np.zeros((n_obs, N))
    H[np.arange(n_obs), obs_idx] = 1.0
    
    # P H^T = X H^T (1/M-1)
    PHt = (X @ H.T) / (M - 1)  # (N, n_obs)
    
    # H P H^T = H X X^T H^T / (M-1)
    HP = (H @ X)  # (n_obs, M)
    HPHt = (HP @ HP.T) / (M - 1)  # (n_obs, n_obs)
    
    # Add observation noise
    R = observation_variance * np.eye(n_obs)
    S = HPHt + R  # (n_obs, n_obs)
    
    # Compute Kalman gain
    # K = PHt S^-1
    try:
        S_inv = np.linalg.inv(S)
        K = PHt @ S_inv  # (N, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = PHt @ S_inv
    
    # Apply localization to the gain
    # For each observed variable j, attenuate the gain for unobserved variables i
    # based on distance
    loc_matrix = np.ones((N, n_obs))
    for j, oj in enumerate(obs_idx):
        for i in range(N):
            # Periodic distance
            d = abs(i - oj)
            d = min(d, N - d)
            loc_matrix[i, j] = np.exp(-0.5 * (d / sigma_loc) ** 2)
    
    K = K * loc_matrix  # Apply localization
    
    # Update ensemble mean
    ens_mean = ens_mean + K @ innovation  # (N,)
    
    # Update ensemble perturbations using LESRF (deterministic)
    # X_new = X - K H X  (in ensemble form)
    # More precisely: X_new = (I - K H) X
    KH = K @ H  # (N, N)
    X_new = X - KH @ X  # (M, N)
    
    # Renormalize perturbations to preserve ensemble variance
    # Scale X_new so that var(X_new, axis=0) matches var(X, axis=0)
    var_old = np.var(X, axis=0, ddof=1)
    var_new = np.var(X_new, axis=0, ddof=1)
    scale = np.sqrt(np.where(var_new > 1e-15, var_old / var_new, 1.0))
    X_new = X_new * scale[np.newaxis, :]
    
    # Construct updated ensemble
    result = ens_mean[np.newaxis, :] + X_new
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, ens_mean[np.newaxis, :])
    
    # Clip to reasonable bounds
    result = np.clip(result, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    # --- Optional: Second pass with refined inflation ---
    # Recompute inflation based on updated ensemble
    obs_ens2 = result[:, obs_idx]
    ens_var_obs2 = np.var(obs_ens2, axis=0, ddof=1)
    obs_mean2 = np.mean(obs_ens2, axis=0)
    innovation2 = observations - obs_mean2
    innov_var2 = np.var(innovation2, ddof=1) if n_obs > 1 else np.mean(innovation2**2)
    expected_innov_var2 = np.mean(ens_var_obs2) + observation_variance
    
    if expected_innov_var2 > 1e-10:
        ratio2 = innov_var2 / expected_innov_var2
        inflation2 = 1.0 + 0.3 * max(0.0, ratio2 - 1.0)
        inflation2 = np.clip(inflation2, 0.9, 1.5)
    else:
        inflation2 = 1.0
    
    # Apply mild additional correction if needed
    if inflation2 > 1.01:
        result[:, obs_idx] = obs_mean2 + inflation2 * (result[:, obs_idx] - obs_mean2)
        result = np.clip(result, -100.0, 100.0)
    
    return result
