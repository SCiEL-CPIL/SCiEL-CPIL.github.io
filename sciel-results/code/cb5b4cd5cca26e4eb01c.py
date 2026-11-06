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


def _lorenz96_propagate(x, F, dt, nsteps):
    """Propagate Lorenz-96 state forward."""
    for _ in range(nsteps):
        x = _rk4_step(x, F, dt)
    return x


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def _compute_localization_matrix(D, observed_indices, localization_length):
    """Compute localization weight matrix for cross-covariance."""
    M = len(observed_indices)
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = _gaussian_localization(dist, localization_length)
    return loc_weights


def _nonlinear_ensemble_smoother(ensemble, obs_vec, observed_indices, observation_variance,
                                  localization_length=4.0):
    """
    Nonlinear ensemble smoother using observation likelihood weighting.
    
    Instead of linear covariance update, uses:
    1. Compute observation likelihood for each ensemble member
    2. Weight ensemble members by likelihood
    3. Use weighted ensemble to estimate mean and covariance
    4. Update unobserved states using weighted regression
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute observation likelihood for each ensemble member
    obs_matrix = ensemble[:, observed_indices]  # (N, M)
    R = observation_variance
    
    # Mahalanobis distance squared for each member
    innovations = obs_vec[None, :] - obs_matrix  # (N, M)
    mahal_dist_sq = np.sum(innovations ** 2, axis=1) / R  # (N,)
    
    # Likelihood weights (normalized)
    log_weights = -0.5 * mahal_dist_sq
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Check if weights are too concentrated (effective sample size)
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is too small, fall back to linear EnKF with localization
    if ess < N * 0.3:
        ens_mean = np.mean(ensemble, axis=0)
        anomalies = ensemble - ens_mean
        obs_anomalies = anomalies[:, observed_indices]
        
        P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
        R_mat = R * np.eye(M)
        S = P_obs + R_mat
        
        PHt = (anomalies.T @ obs_anomalies) / (N - 1)
        
        loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
        PHt_loc = PHt * loc_weights
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        updated = ensemble.copy()
        for m in range(N):
            innovation = obs_vec - ensemble[m, observed_indices]
            Si_innov = S_inv @ innovation
            delta_x = PHt_loc @ Si_innov
            updated[m] += delta_x
        
        # Inflation
        obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
        obs_spread_pre = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
        obs_noise_std = np.sqrt(R)
        mean_ens_spread = np.mean(obs_spread_pre)
        
        if mean_ens_spread < obs_noise_std * 0.5:
            inflation = 1.1
        else:
            inflation = 1.0
        
        if inflation > 1.0:
            updated_mean = np.mean(updated, axis=0)
            updated = updated_mean + inflation * (updated - updated_mean)
        
        return updated
    
    # Nonlinear update: use weighted ensemble
    weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
    
    weighted_anomalies = ensemble - weighted_mean[None, :]
    weighted_cov = (weighted_anomalies.T * weights[None, :]) @ weighted_anomalies / (np.sum(weights) - 1)
    
    obs_anomalies = weighted_anomalies[:, observed_indices]
    P_obs = (obs_anomalies.T * weights[None, :]) @ obs_anomalies / (np.sum(weights) - 1)
    R_mat = R * np.eye(M)
    S = P_obs + R_mat
    
    PHt = (weighted_anomalies.T * weights[None, :]) @ obs_anomalies / (np.sum(weights) - 1)
    
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv  # (D, M)
    
    updated = ensemble.copy()
    for m in range(N):
        innovation = obs_vec - ensemble[m, observed_indices]
        delta_x = K @ innovation
        updated[m] += delta_x
    
    # Resample if needed to maintain diversity
    if ess < N * 0.5:
        cumulative_weights = np.cumsum(weights)
        cumulative_weights /= cumulative_weights[-1]
        random_start = np.random.uniform(0, 1.0 / N)
        resample_indices = np.searchsorted(cumulative_weights, random_start + np.arange(N) / N)
        resample_indices = np.clip(resample_indices, 0, N - 1)
        updated = updated[resample_indices]
    
    # Inflation
    obs_mean_pre = np.mean(ensemble[:, observed_indices], axis=0)
    obs_spread_pre = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
    obs_noise_std = np.sqrt(R)
    mean_ens_spread = np.mean(obs_spread_pre)
    
    if mean_ens_spread < obs_noise_std * 0.5:
        inflation = 1.1
    else:
        inflation = 1.0
    
    if inflation > 1.0:
        updated_mean = np.mean(updated, axis=0)
        updated = updated_mean + inflation * (updated - updated_mean)
    
    return updated


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
    deadline = 145.0
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    try:
        result = _nonlinear_ensemble_smoother(
            ensemble, obs_vec, observed_indices, observation_variance,
            localization_length=4.0
        )
    except Exception:
        ens_mean = np.mean(ensemble, axis=0)
        anomalies = ensemble - ens_mean
        obs_anomalies = anomalies[:, observed_indices]
        P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
        R = observation_variance * np.eye(M)
        S = P_obs + R
        PHt = (anomalies.T @ obs_anomalies) / (N - 1)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        result = ensemble.copy()
        for m in range(N):
            innovation = obs_vec - ensemble[m, observed_indices]
            Si_innov = S_inv @ innovation
            delta_x = PHt @ Si_innov
            result[m] += delta_x
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
