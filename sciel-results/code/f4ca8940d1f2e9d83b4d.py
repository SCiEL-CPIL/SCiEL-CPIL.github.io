"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        ip1 = (i + 1) % n
        im1 = (i - 1) % n
        im2 = (i - 2) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance,
                            localization_length=4.0, inflation_factor=1.05,
                            max_inflation=1.5):
    """
    Ensemble Kalman Filter update with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    inflation_factor : float - base multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute observation-space covariance P_obs
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    
    # Add observation noise
    R = observation_variance * np.eye(M)
    S = P_obs + R  # Innovation covariance (M, M)
    
    # Compute cross-covariance PHt
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Apply localization to the cross-covariance
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            # Minimum image distance on the ring
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    
    # Apply localization to cross-covariance
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Solve for S_inv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Compute updated ensemble
    # For each member m:
    # innovation_m = obs_vec - ensemble[m, observed_indices]
    # delta_x_m = PHt_loc @ S_inv @ innovation_m
    updated_ensemble = ensemble.copy()
    
    for m in range(N):
        innovation = obs_vec - ensemble[m, observed_indices]  # (M,)
        Si_innov = S_inv @ innovation  # (M,)
        delta_x = PHt_loc @ Si_innov  # (D,)
        updated_ensemble[m] += delta_x
    
    # Adaptive inflation
    # Compute pre-inflation ensemble mean in observation space
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    obs_spread_pre = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
    
    # Compute innovation magnitude (using the single obs vector)
    # Innovation for the mean: obs_vec - obs_mean_pre
    innov_mean = obs_vec - obs_mean_pre
    # We can estimate the "expected" innovation spread from the ensemble spread
    # If the ensemble spread is too small compared to the observation noise or innovation, inflate.
    # A common metric: compare std(innovations) to std(ensemble obs spread).
    # Here we don't have a distribution of innovations, just one.
    # Heuristic: If the ensemble spread is very small, inflate.
    # Let's use the ratio of obs noise std to ensemble spread as a proxy for under-dispersion.
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    if mean_ens_spread < obs_noise_std * 0.5:
        # Under-dispersed
        inflation = 1.1
    else:
        inflation = 1.0
    
    # Apply inflation about the ensemble mean
    if inflation > 1.0:
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
        # If it's (N, M), assume all rows are the same or take the first one
        # Standard EnKF uses a single observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Basic EKF with localization and inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=4.0,
        inflation_factor=1.05,
        max_inflation=1.5
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
