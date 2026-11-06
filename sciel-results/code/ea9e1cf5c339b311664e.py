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


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance, loc_length=5.0):
    """
    Ensemble Kalman Filter with optimal gain (regression-based) and adaptive inflation.
    Uses minimum-image distance localization on the ring.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # Ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered anomalies
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute P_obs = cov(observed) + R
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance P_xH^T
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localization: minimum-image distance on ring
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = loc_length
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Compute S_inv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Innovation for each member
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Gain-like update: delta_x = PHt_loc @ S_inv @ innovation
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta_x = Si_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta_x  # (N, D)
    
    # Adaptive inflation: compare ensemble spread to observation noise
    ens_spread_obs = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    
    if mean_spread < obs_noise_std * 0.7:
        inflation = 1.08
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    elif mean_spread > obs_noise_std * 2.0:
        inflation = 0.97
        updated_mean = np.mean(updated_ensemble, axis=0)
        updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    # Safety: clip and handle NaNs
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _ensemble_kalman_smoother(ensemble, obs_vec, observed_indices, observation_variance, F=8.0):
    """
    Enhanced ensemble Kalman filter with multiple refinement passes.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # First pass: standard EKF with localization L=5
    result = _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance, loc_length=5.0)
    
    # Second pass: re-apply EKF with slightly different localization
    # to smooth out any inconsistencies
    if N > 4:
        # Use a wider localization for the second pass
        ens_mean = np.mean(result, axis=0)
        anomalies = result - ens_mean
        obs_anomalies = anomalies[:, observed_indices]
        
        P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
        R = observation_variance * np.eye(M)
        S = P_obs + R
        
        PHt = (anomalies.T @ obs_anomalies) / (N - 1)
        
        # Wider localization
        d_idx = np.arange(D)
        loc_weights = np.ones((D, M))
        L = 7.0
        for m_idx, obs_i in enumerate(observed_indices):
            dist = np.abs(d_idx - obs_i)
            dist = np.minimum(dist, D - dist)
            loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
        
        PHt_loc = PHt * loc_weights
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        ens_obs = result[:, observed_indices]
        innovations = obs_vec[None, :] - ens_obs
        Si_innov = innovations @ S_inv.T
        delta_x = Si_innov @ PHt_loc.T
        
        # Blend the two updates
        result = result + 0.5 * delta_x
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result


def _local_dynamics_refinement(ensemble, obs_vec, observed_indices, observation_variance, F=8.0, dt=0.01, n_steps=5):
    """
    Refine ensemble members using short dynamics propagation to maintain consistency.
    For each member, propagate forward briefly and blend back with the current state.
    """
    N, D = ensemble.shape
    result = ensemble.copy()
    
    for i in range(N):
        # Propagate this member forward n_steps
        x = result[i].copy()
        for step in range(n_steps):
            x = _rk4_step(x, F, dt)
        
        # Blend: keep 70% of current, 30% of propagated
        # This helps maintain dynamical consistency without over-correcting
        result[i] = 0.7 * result[i] + 0.3 * x
    
    # Re-apply a light EKF update to pull back toward observations
    result = _ensemble_kalman_filter(result, obs_vec, observed_indices, observation_variance, loc_length=6.0)
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result


def _adaptive_inflation(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Compute adaptive inflation factor based on ensemble spread vs observation noise.
    """
    N, D = ensemble.shape
    ens_obs = ensemble[:, observed_indices]
    ens_spread = np.std(ens_obs, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread)
    
    if mean_spread < obs_noise_std * 0.5:
        return 1.12
    elif mean_spread < obs_noise_std * 0.8:
        return 1.06
    elif mean_spread > obs_noise_std * 2.5:
        return 0.95
    elif mean_spread > obs_noise_std * 1.5:
        return 0.98
    return 1.0


def _apply_inflation(ensemble, factor):
    """Apply multiplicative inflation around the ensemble mean."""
    if abs(factor - 1.0) < 1e-6:
        return ensemble
    mean = np.mean(ensemble, axis=0)
    return mean + factor * (ensemble - mean)


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
    deadline = start_time + 110.0  # safety margin
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Step 1: Primary EKF update with moderate localization
    result = _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance, loc_length=5.0)
    
    # Check deadline
    if time.time() > deadline - 5.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Step 2: Adaptive inflation
    infl_factor = _adaptive_inflation(result, obs_vec, observed_indices, observation_variance)
    result = _apply_inflation(result, infl_factor)
    
    # Check deadline
    if time.time() > deadline - 3.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Step 3: Second EKF pass with wider localization for smoothing
    result = _ensemble_kalman_smoother(result, obs_vec, observed_indices, observation_variance)
    
    # Check deadline
    if time.time() > deadline - 2.0:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Step 4: Light dynamics-based refinement to maintain consistency
    result = _local_dynamics_refinement(result, obs_vec, observed_indices, observation_variance, F=8.0, dt=0.01, n_steps=3)
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
