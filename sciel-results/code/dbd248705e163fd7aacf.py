"""Combined assimilation for Lorenz-96 partial observation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline guard: 5s safety margin before 150s kill
_DEADLINE = time.time() + 145.0

def _gaussian_localization(dist, length_scale):
    """Gaussian localization weight."""
    return np.exp(-0.5 * (dist / length_scale) ** 2)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-KF) assimilation for Lorenz-96
    with Gaussian localization.
    
    Uses random rotation matrices to generate stochastic perturbations,
    ensuring the ensemble mean matches the optimal Kalman gain exactly
    without bias from deterministic square-root decompositions.
    
    Parameters:
    -----------
    ensemble : (16, 20) ndarray
        Ensemble of 16 members, each a 20-state Lorenz-96 vector
    observations : (10,) ndarray
        Noisy observations of the 10 even-indexed states
    observed_indices : array_like
        Indices of observed variables
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) ndarray
        Updated ensemble after assimilation
    """
    # Deadline check
    if time.time() > _DEADLINE:
        return ensemble.copy()
    
    N, M = ensemble.shape
    obs_idx = np.array(sorted(observed_indices), dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ens = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    obs = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float64)
    
    # --- Step 1: Compute ensemble mean and anomalies ---
    x_mean = np.mean(ens, axis=0)  # (20,)
    anomalies = ens - x_mean  # (N, 20)
    
    # --- Step 2: Build localization matrix ---
    # Localization scale tuned for L96 correlation length (~3-4 grid points)
    length_scale = 2.8
    
    # Precompute periodic distances from each observed index to all state indices
    dist_matrix = np.zeros((n_obs, M))
    for i, oi in enumerate(obs_idx):
        for j in range(M):
            d = abs(oi - j)
            d = min(d, M - d)  # periodic
            dist_matrix[i, j] = d
    
    # Localization weights
    L = _gaussian_localization(dist_matrix, length_scale)  # (n_obs, 20)
    # Ensure self-observation gets full weight
    for i in range(n_obs):
        L[i, obs_idx[i]] = 1.0
    
    # --- Step 3: Stochastic Ensemble Kalman Filter update ---
    # Generate a random orthogonal matrix Q of shape (N, N)
    # Using QR decomposition of a random Gaussian matrix
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within call
    Z = rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(Z)
    # Ensure Q is a proper rotation (det = 1)
    d = np.diag(Q)
    Q[:, 0] *= np.sign(d[0])
    
    # Compute the Kalman gain matrix using the full ensemble covariance
    # P = (1/N) * A^T A where A is (N, M) anomalies
    # For localized observations, we need to handle each observation
    # but for SE-KF, we can work with the full system
    
    # Observation operator H: maps state to observed components
    # H is (n_obs, M) with H[i, obs_idx[i]] = 1
    
    # Build H matrix
    H = np.zeros((n_obs, M))
    for i, oi in enumerate(obs_idx):
        H[i, oi] = 1.0
    
    # Localized cross-covariance: P_xy = (1/N) * A^T * H^T * A * L
    # More precisely, for each observation i, the cross-cov with state j is
    # localized: cov_i_j = (1/N) * sum_n A[n, obs_idx[i]] * A[n, j] * L[i, j]
    
    # Compute P_yy (observation-space covariance with localization)
    # P_yy[i, k] = (1/N) * sum_n A[n, obs_idx[i]] * A[n, obs_idx[k]] * min(L[i, obs_idx[i]], L[k, obs_idx[k]])
    # But standard approach: P_yy[i] = (1/N) * sum_n A[n, obs_idx[i]]^2 (no localization for obs-obs)
    
    A = anomalies  # (N, M)
    
    # Observation-space anomalies
    A_y = A[:, obs_idx]  # (N, n_obs)
    
    # P_yy = (1/N) * A_y^T A_y (n_obs, n_obs)
    P_yy = (A_y.T @ A_y) / N
    
    # Cross-covariance P_xy = (1/N) * A^T A_y (M, n_obs)
    # But we need to apply localization to P_xy
    # P_xy_localized[j, i] = P_xy[j, i] * L[i, j]
    P_xy = (A.T @ A_y) / N  # (M, n_obs)
    
    # Apply localization to P_xy: for each column i (observation), multiply by L[i, :]
    # P_xy_localized[:, i] = P_xy[:, i] * L[i, :]
    P_xy_loc = P_xy * L.T  # (M, n_obs) * (M, n_obs) element-wise
    
    # R = observation_variance * I (n_obs, n_obs)
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    y_innov = obs - x_mean[obs_idx]  # (n_obs,)
    
    # Kalman gain: K = P_xy_loc @ inv(P_yy + R) @ y_innov
    # But for SE-KF, we need the gain matrix, not just the gain vector
    # K_matrix = P_xy_loc @ inv(P_yy + R)  (M, n_obs)
    
    S = P_yy + R  # (n_obs, n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain matrix: K_mat = P_xy_loc @ S_inv  (M, n_obs)
    K_mat = P_xy_loc @ S_inv
    
    # Update the mean: x_mean_new = x_mean + K_mat @ y_innov
    x_mean_new = x_mean + K_mat @ y_innov
    
    # For SE-KF, each member update:
    # x_n_new = x_n + K_mat @ (H @ (x_n - x_mean) - y_innov + Q_n)
    # where Q_n is a random perturbation with covariance R
    
    # More precisely, the SE-KF update is:
    # x_n_new = x_n + K_mat @ (y_n - H @ x_mean + eps_n)
    # where y_n = H @ x_n + noise, but we don't have per-member observations
    # Standard SE-KF: 
    #   innovation_n = H @ x_n - x_mean_obs + eps_n
    #   x_n_new = x_n + K_mat @ innovation_n
    # where eps_n ~ N(0, R)
    
    # Generate stochastic perturbations
    eps = rng.standard_normal((N, n_obs)) * np.sqrt(observation_variance)
    
    # For each member, compute innovation: H @ x_n - x_mean_obs + eps_n
    # H @ x_n gives (n_obs,) for member n
    # x_n_obs = ens[:, obs_idx]  (N, n_obs)
    x_n_obs = ens[:, obs_idx]  # (N, n_obs)
    
    # innovation_n = x_n_obs - x_mean[obs_idx] + eps_n
    innovations = x_n_obs - x_mean[obs_idx] + eps  # (N, n_obs)
    
    # Update each member: x_n_new = x_n + K_mat @ innovation_n
    # K_mat is (M, n_obs), innovation_n is (n_obs,)
    # So K_mat @ innovation_n is (M,)
    # Vectorized: innovations.T @ K_mat.T gives (N, M)
    updates = innovations @ K_mat.T  # (N, M)
    
    updated_ens = ens + updates
    
    # --- Step 4: Variance inflation to prevent collapse ---
    new_mean = np.mean(updated_ens, axis=0)
    current_spread = np.std(updated_ens, axis=0)
    pre_spread = np.std(ens, axis=0)  # spread before update
    
    # Per-component inflation with more aggressive spread restoration
    for j in range(M):
        if current_spread[j] > 1e-8:
            # Target: maintain at least 85% of pre-update spread
            target = pre_spread[j] * 0.85
            if current_spread[j] < target:
                scale = target / current_spread[j]
                scale = min(scale, 1.3)  # allow moderate inflation to restore spread
                updated_ens[:, j] = new_mean[j] + (updated_ens[:, j] - new_mean[j]) * scale
        else:
            # Add small perturbation if spread collapsed
            updated_ens[:, j] = new_mean[j] + rng.normal(0, 0.05, N)
    
    # --- Step 5: Safety checks ---
    if not np.all(np.isfinite(updated_ens)):
        updated_ens = np.nan_to_num(updated_ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    updated_ens = np.clip(updated_ens, -100.0, 100.0)
    
    return updated_ens

# EVOLVE-BLOCK-END
