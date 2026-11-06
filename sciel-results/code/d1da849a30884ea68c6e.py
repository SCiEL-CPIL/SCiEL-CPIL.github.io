"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute localization weights matrix of shape (D, M).
    D = state dimension, M = number of observations.
    Uses minimum-image distance on the ring.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return weights


def _nonlinear_particle_update(ensemble, obs_vec, obs_idx, obs_var, D, M, loc_width, blend_alpha=0.4):
    """
    Nonlinear particle filter step using importance resampling and KDE.
    Blends with linear EnKF for stability.
    """
    N = ensemble.shape[0]
    
    # Compute linear EnKF update first
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    obs_anomalies = anomalies[:, obs_idx]
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Eigenvalue-based regularization
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv
    
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas_linear = innovations @ K.T
    updated_linear = ensemble + deltas_linear
    
    # Nonlinear particle filter component
    # Compute importance weights based on observation likelihood
    # Using a Gaussian likelihood with the ensemble's observation-space covariance
    
    # Observation-space innovation for each particle
    particle_innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    
    # Compute log-likelihood weights
    # w_i ∝ exp(-0.5 * (y - Hx_i)^T S^{-1} (y - Hx_i))
    # But we want to capture non-Gaussian structure, so use a bandwidth-adapted kernel
    
    # Adaptive bandwidth based on ensemble spread in observation space
    obs_spread = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    bandwidth = np.maximum(obs_spread, np.sqrt(obs_var) * 0.5)
    
    # Compute kernel density estimate-based weights
    # For each particle, compute the likelihood using a product of 1D kernels
    log_weights = np.zeros(N)
    for i in range(N):
        diff = particle_innovations[i]  # (M,)
        # Gaussian kernel with adaptive bandwidth
        log_w = -0.5 * np.sum((diff / bandwidth) ** 2)
        log_weights[i] = log_w
    
    # Normalize weights
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Weighted mean in observation space
    weighted_obs_mean = np.sum(ensemble[:, obs_idx] * weights[:, None], axis=0)
    
    # Compute the nonlinear correction: adjust particles toward the weighted mean
    # but preserve the ensemble structure
    # The correction is: delta_nl_i = (weighted_obs_mean - ensemble[i, obs_idx]) * K_nl
    # where K_nl is a nonlinear gain
    
    # For the nonlinear gain, use the cross-covariance but with localization
    # and a damping factor to avoid over-correction
    nonlinear_gain = PHt_loc * 0.5  # Damp the nonlinear correction
    
    # Nonlinear innovations: difference between weighted mean and each particle
    nl_innovations = (weighted_obs_mean[None, :] - ensemble[:, obs_idx])  # (N, M)
    deltas_nonlinear = nl_innovations @ nonlinear_gain.T
    
    # Blend linear and nonlinear updates
    updated_nonlinear = ensemble + deltas_nonlinear
    updated = (1.0 - blend_alpha) * updated_linear + blend_alpha * updated_nonlinear
    
    return updated, weights


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check time before expensive computation
    if time.time() > deadline - 5.0:
        # Return a simple mean-preserving update as fallback
        return ensemble.copy()
    
    # Precompute localization matrix
    L_loc = 4.0
    loc_mat = _compute_localization_matrix(D, obs_idx, L_loc)
    
    # Ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # Observation-space anomalies
    obs_anomalies = anomalies[:, obs_idx]
    
    # Covariance matrices
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_obs + R
    
    # Eigenvalue-based regularization
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    # Cross-covariance with localization
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    # Solve for S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain
    K = PHt_loc @ S_inv
    
    # Linear EnKF update
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas_linear = innovations @ K.T
    updated_linear = ensemble + deltas_linear
    
    # Check time
    if time.time() > deadline - 3.0:
        # Return linear update as fallback
        updated_linear = np.nan_to_num(updated_linear, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_linear = np.clip(updated_linear, -100.0, 100.0)
        return updated_linear
    
    # Nonlinear particle filter component
    # Compute importance weights based on observation likelihood
    particle_innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    
    # Adaptive bandwidth based on ensemble spread in observation space
    obs_spread = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    bandwidth = np.maximum(obs_spread, np.sqrt(observation_variance) * 0.5)
    
    # Compute kernel density estimate-based weights
    log_weights = np.zeros(N)
    for i in range(N):
        diff = particle_innovations[i]  # (M,)
        log_w = -0.5 * np.sum((diff / bandwidth) ** 2)
        log_weights[i] = log_w
    
    # Normalize weights
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Weighted mean in observation space
    weighted_obs_mean = np.sum(ensemble[:, obs_idx] * weights[:, None], axis=0)
    
    # Nonlinear correction with damped gain
    nonlinear_gain = PHt_loc * 0.4  # Damp the nonlinear correction
    nl_innovations = (weighted_obs_mean[None, :] - ensemble[:, obs_idx])
    deltas_nonlinear = nl_innovations @ nonlinear_gain.T
    
    # Blend linear and nonlinear updates
    blend_alpha = 0.35
    updated = (1.0 - blend_alpha) * updated_linear + blend_alpha * (ensemble + deltas_nonlinear)
    
    # Adaptive inflation based on innovation statistics
    obs_spread_pre = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    if mean_ens_spread < obs_noise_std * 0.6:
        inflation = 1.08
    elif mean_ens_spread < obs_noise_std * 0.8:
        inflation = 1.03
    else:
        inflation = 1.0
    
    if inflation > 1.0:
        updated = ens_mean + inflation * (updated - ens_mean)
    
    # Final safety checks
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated

# EVOLVE-BLOCK-END
