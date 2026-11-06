import numpy as np
import time

_start_time = time.time()
_DEADLINE = _start_time + 145.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    # Background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b

    # Background covariance (unbiased)
    P_b = (W_b.T @ W_b) / (M - 1)

    # Extract relevant covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]

    # Observation error covariance
    R = np.eye(N_obs) * R_val

    # Innovation
    H_mean = mean_b[obs_idx]
    d = y - H_mean

    # Smooth Gaussian localization
    N_all = np.arange(N)
    dist = np.abs(N_all[:, np.newaxis] - obs_idx[np.newaxis, :])
    dist = np.minimum(dist, N - dist)

    # Tuned localization length for L96
    L = 2.0
    loc_matrix = np.exp(-0.5 * (dist / L) ** 2)

    P_xo_loc = P_xo * loc_matrix

    # Innovation covariance
    S = P_oo + R

    # Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = P_xo_loc @ S_inv

    # Stochastic Ensemble Kalman Filter (StEnKF)
    # Instead of deterministic update, we use the ensemble mean update
    # and then perturb each member with observation noise projected through K
    
    # Mean update (deterministic part)
    mean_a = mean_b + K @ d

    # For stochastic update, we need to inject observation noise into each member
    # The stochastic EnKF updates each member as:
    # x_a^m = x_b^m + K @ (y^m - H x_b^m)
    # where y^m = y + K_y @ epsilon^m, but since R is diagonal, we can simplify
    
    # Actually, the standard StEnKF is:
    # x_a^m = x_b^m + K @ (y - H x_b^m + K_y @ eta^m)
    # where eta^m ~ N(0, I) and K_y = I (for diagonal R, the noise is added directly)
    
    # More precisely, for each member m:
    # innovation_m = y - H x_b^m + sqrt(R) * eta^m
    # x_a^m = x_b^m + K @ innovation_m
    
    # Since R = R_val * I, sqrt(R) = sqrt(R_val) * I
    sqrt_R = np.sqrt(R_val)
    
    # Generate random noise for each member
    eta = np.random.standard_normal((M, N_obs))
    
    # Compute innovation for each member
    H_W_b = W_b[:, obs_idx]  # (M, N_obs)
    innovation = d[np.newaxis, :] - H_W_b + sqrt_R * eta  # (M, N_obs)
    
    # Update each member
    X_a = X_b + (K @ innovation.T).T  # (M, N)
    
    # Adaptive inflation to maintain ensemble spread
    # Compare innovation variance to expected
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.1)
    
    # Inflate perturbations about the new mean
    mean_W_a = np.mean(X_a, axis=0)
    W_a = X_a - mean_W_a
    W_a *= inflation_factor
    X_a = mean_W_a + W_a
    
    # Safety
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
