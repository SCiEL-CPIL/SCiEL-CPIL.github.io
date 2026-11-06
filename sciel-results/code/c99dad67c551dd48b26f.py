"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    n = len(x)
    # Periodic indices
    x_ip1 = np.roll(x, -1)  # x_{i+1}
    x_im1 = np.roll(x, 1)   # x_{i-1}
    x_im2 = np.roll(x, 2)   # x_{i-2}
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _ensemble_square_root_filter(ensemble, obs_vec, observed_indices, observation_variance,
                                  localization_length=3.5, min_inflation=1.0, max_inflation=1.2):
    """
    Ensemble Square Root Filter (ESRF) with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    min_inflation : float - minimum multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs (M, M)
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt (D, M)
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    
    # Apply localization to the cross-covariance
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            # Minimum image distance on the ring
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = np.exp(-0.5 * (dist / localization_length) ** 2)
    
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv using Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        S_inv = np.linalg.inv(L.T @ L)
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    # Compute the gain matrix K = PHt_loc @ S_inv (D, M)
    K = PHt_loc @ S_inv
    
    # Apply the update to each ensemble member
    # innovation_m = obs_vec - ensemble[m, observed_indices]
    # delta_x_m = K @ innovation_m
    updated_ensemble = ensemble.copy()
    
    # Vectorized innovation computation
    obs_ens = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - obs_ens  # (N, M)
    
    # delta_x = innovations @ K.T  (N, M) @ (M, D) = (N, D)
    delta_x = innovations @ K.T
    updated_ensemble += delta_x
    
    # Adaptive inflation based on ensemble spread vs observation spread
    # Compute the ensemble spread in observation space
    obs_spread = np.std(ensemble[:, observed_indices], axis=0, ddof=1)  # (M,)
    obs_noise_std = np.sqrt(observation_variance)
    
    # Compare ensemble spread to expected spread from observations
    # If ensemble spread is too small, inflate
    mean_ens_spread = np.mean(obs_spread)
    
    # Target: ensemble spread should be at least comparable to observation noise
    if mean_ens_spread < obs_noise_std * 0.7:
        # Under-dispersed: inflate more
        inflation = min(max_inflation, 1.0 + (obs_noise_std - mean_ens_spread) / max(obs_noise_std, 1e-10) * 0.3)
    elif mean_ens_spread > obs_noise_std * 1.5:
        # Over-dispersed: slight deflation
        inflation = max(min_inflation, 1.0 - (mean_ens_spread - obs_noise_std) / max(mean_ens_spread, 1e-10) * 0.1)
    else:
        inflation = 1.0
    
    # Apply inflation about the ensemble mean
    if inflation != 1.0:
        updated_ensemble = ens_mean + inflation * (updated_ensemble - ens_mean)
    
    # Ensure finite values and clip to valid range
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


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
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        # If it's (N, M), take the first row as the observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure we have the right number of observations
    if len(obs_vec) != M:
        # Fallback: use what we have
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            # Pad with zeros if needed
            obs_vec = np.zeros(M)
            obs_vec[:len(obs_vec)] = obs_vec
    
    # Basic ESRF with localization and inflation
    result = _ensemble_square_root_filter(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.5,
        min_inflation=1.0,
        max_inflation=1.2
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard: if we're running long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 145.0:  # Safety margin before 150s limit
        return result
    
    return result

# EVOLVE-BLOCK-END
