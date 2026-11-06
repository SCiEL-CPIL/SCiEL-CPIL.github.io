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


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, D, M, loc_width):
    """
    Standard EnKF update with localization.
    Returns updated ensemble of shape (N, D).
    """
    N = ensemble.shape[0]
    
    # Precompute localization matrix
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)  # (D, M)
    
    # Ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation-space anomalies
    obs_anomalies = anomalies[:, obs_idx]  # (N, M)
    
    # Covariance matrices
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Regularize S for numerical stability
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    # Cross-covariance with localization
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    PHt_loc = PHt * loc_mat  # (D, M)
    
    # Solve for S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain
    K = PHt_loc @ S_inv  # (D, M)
    
    # Update all members at once (vectorized)
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    deltas = innovations @ K.T  # (N, D)
    updated_ensemble = ensemble + deltas
    
    return updated_ensemble


def _adaptive_inflation(ensemble, obs_vec, obs_idx, obs_var, D):
    """
    Compute adaptive multiplicative inflation factor based on innovation statistics.
    Returns inflation factor (scalar >= 1.0).
    """
    N = ensemble.shape[0]
    M = len(obs_idx)
    
    # Compute ensemble spread in observation space
    obs_spread = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    obs_noise_std = np.sqrt(obs_var)
    
    # Innovation statistics
    ens_mean_obs = np.mean(ensemble[:, obs_idx], axis=0)
    innovations = obs_vec - ens_mean_obs
    innovation_norm = np.sqrt(np.sum(innovations ** 2) / M)
    expected_innovation = np.sqrt(np.mean(obs_spread ** 2) + obs_var)
    
    # Ratio of actual to expected innovation
    ratio = innovation_norm / max(expected_innovation, 1e-10)
    
    # Adaptive inflation: if innovations are too large, inflate
    if ratio > 1.2:
        inflation = 1.0 + 0.05 * (ratio - 1.0)
    elif ratio > 1.0:
        inflation = 1.0 + 0.02 * (ratio - 1.0)
    elif ratio < 0.7:
        # Under-dispersive: more aggressive inflation
        inflation = 1.0 + 0.08 * (0.7 - ratio)
    else:
        inflation = 1.0
    
    # Cap inflation
    inflation = np.clip(inflation, 1.0, 1.15)
    
    return inflation


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
    
    # Two-pass EnKF: first narrow localization, then wider
    # Pass 1: narrow localization to handle local nonlinearities
    loc_width_narrow = 3.0
    updated_narrow = _enkf_update(ensemble, obs_vec, obs_idx, observation_variance, D, M, loc_width_narrow)
    
    # Check time
    if time.time() > deadline - 5.0:
        updated_ensemble = np.nan_to_num(updated_narrow, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble
    
    # Pass 2: wider localization for global consistency
    loc_width_wide = 5.0
    updated_wide = _enkf_update(updated_narrow, obs_vec, obs_idx, observation_variance, D, M, loc_width_wide)
    
    # Check time
    if time.time() > deadline - 3.0:
        updated_ensemble = np.nan_to_num(updated_wide, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
        return updated_ensemble
    
    # Adaptive inflation
    inflation = _adaptive_inflation(updated_wide, obs_vec, obs_idx, observation_variance, D)
    
    if inflation > 1.0:
        ens_mean = np.mean(updated_wide, axis=0)
        updated_wide = ens_mean + inflation * (updated_wide - ens_mean)
    
    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_wide, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
