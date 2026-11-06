"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble square-root Kalman filter with localization and adaptive inflation
    for Lorenz-96 state reconstruction from partial observations.
    """
    t_start = time.time()
    DEADLINE = 148.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M ensemble members, N state dimension
    n_obs = len(obs_idx)

    # Background mean
    mean_b = X.mean(axis=0)

    # Anomalies
    W = X - mean_b  # M x N

    # Localization: Gaussian taper with circular distance
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    range_param = 3.0
    taper = np.exp(-0.5 * (dist / range_param) ** 2)

    # Localized background covariance
    P_b = (W.T @ W) / (M - 1)
    P_loc = P_b * taper

    # Observation-space covariance
    H_idx = obs_idx
    P_obs = P_loc[np.ix_(H_idx, H_idx)] + R * np.eye(n_obs)

    # Innovation
    innovation = y - mean_b[H_idx]

    # Compute Kalman gain via Cholesky
    try:
        L = np.linalg.cholesky(P_obs)
        # K = P_loc[:, H_idx] @ P_obs^{-1}
        # Using Cholesky: K = (P_loc[:, H_idx] @ L^{-T}) @ L^{-1}
        temp = np.linalg.solve(L, P_loc[:, H_idx].T)  # n_obs x N
        K = np.linalg.solve(L.T, temp).T  # N x n_obs
    except np.linalg.LinAlgError:
        K = P_loc[:, H_idx] @ np.linalg.pinv(P_obs)

    # Updated mean
    mean_a = mean_b + K @ innovation

    # Ensemble square-root update:
    # X_a = mean_a + (W + K @ (H W - innovation)) / something
    # Standard EnKF: X_a[:,i] = mean_a + W[:,i] + K @ (H W[:,i] - innovation)
    # But this is the "perturbed observation" form. For square root:
    # X_a = mean_a + W * (I - 0.5*K*H) ... not quite.
    
    # Use the standard ensemble Kalman filter update:
    # For each member: X_a_i = X_b_i + K @ (y - H X_b_i)
    # But to preserve the mean, we center first.
    
    HW = W[:, H_idx]  # M x n_obs
    innovation_members = innovation[None, :] - HW  # M x n_obs
    
    # Update
    X_a = X + (innovation_members @ K.T)  # M x N
    
    # Re-center to exact mean
    X_a = X_a - X_a.mean(axis=0) + mean_a

    # Adaptive inflation based on innovation statistics
    # Compare innovation variance to expected
    ens_obs_var = W[:, H_idx].var(axis=0)
    expected_innov_var = ens_obs_var + R
    actual_innov_var = innovation.var()
    
    if expected_innov_var.mean() > 1e-10 and actual_innov_var > 1e-10:
        inflation_factor = np.sqrt(actual_innov_var / expected_innov_var.mean())
        inflation_factor = np.clip(inflation_factor, 1.0, 1.2)
    else:
        inflation_factor = 1.0

    X_a = mean_a + inflation_factor * (X_a - mean_a)

    # Dynamical consistency: one step of Lorenz-96 to improve physical consistency
    # This is a soft correction - blend with the dynamical forecast
    # Compute one-step Lorenz-96 for each member
    def lorenz96_step(x, dt=0.01):
        """One RK4 step of Lorenz-96."""
        n = len(x)
        F = 8.0  # Use a reasonable F; but we don't know F exactly
        # Actually we don't know F. Skip dynamical step if uncertain.
        # Instead, do a simple smoothing based on neighboring values
        return x

    # Since we don't know F, skip the dynamical correction.
    # Instead, apply a simple spatial smoothing to reduce ensemble spread artifacts
    # Only for unobserved variables
    
    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
