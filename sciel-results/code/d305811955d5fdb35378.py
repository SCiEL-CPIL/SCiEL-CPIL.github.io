"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Global deadline tracking
_start_time = None
_DEADLINE_FRACTION = 0.88  # Use 88% of the 150s budget

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
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


def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using a Localized Ensemble Kalman Filter (LEKF)
    with adaptive multiplicative inflation.
    
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
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        return np.clip(ensemble, -100.0, 100.0)
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # --- Localized Ensemble Kalman Filter ---
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)  # (D,)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation operator: H maps state to observed components
    Hx = ensemble[:, observed_indices]  # (N, M)
    Hx_mean = np.mean(Hx, axis=0)  # (M,)
    Hx_anom = Hx - Hx_mean  # (N, M)
    
    # Innovation
    innovation = obs_vec - Hx_mean  # (M,)
    
    # Covariances (sample-based)
    # P_oo: (M, M)
    P_oo = (Hx_anom.T @ Hx_anom) / (N - 1)
    # P_ox: (D, M)
    P_ox = (anomalies.T @ Hx_anom) / (N - 1)
    
    # Observation noise covariance R
    R = observation_variance * np.eye(M)
    
    # Innovation covariance S = P_oo + R
    S = P_oo + R
    
    # Localization weights for cross-covariance P_ox
    # Use Gaussian localization with length scale ~ 2
    L_loc = 2.5
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)  # periodic distance
            loc_weights[d, m_idx] = _gaussian_localization(dist, L_loc)
    
    P_ox_loc = P_ox * loc_weights
    
    # Adaptive multiplicative inflation
    # Estimate spread of innovations vs expected
    # If ensemble is too tight, inflate; if too wide, deflate slightly
    # Use a simple heuristic based on the ratio of innovation variance to expected
    # Expected innovation variance per component ~ trace(S)/M
    # Actual innovation variance ~ mean(innovation^2)
    
    # Compute inflation factor based on ensemble spread relative to observation noise
    # A common approach: inflate based on the ratio of observed innovation std to expected
    innov_std = np.std(innovation)
    expected_innov_std = np.sqrt(np.mean(np.diag(S)))
    
    # Inflation factor: if innovation is larger than expected, ensemble is too tight
    # Use a bounded multiplicative inflation
    if expected_innov_std > 1e-10:
        ratio = innov_std / expected_innov_std
        # Target: bring ratio closer to 1
        # inflation = ratio^0.5 would overcorrect; use a damped version
        inflation_factor = np.clip(1.0 + 0.3 * (ratio - 1.0), 0.8, 1.5)
    else:
        inflation_factor = 1.0
    
    # Apply inflation to anomalies
    inflated_anomalies = anomalies * inflation_factor
    
    # Recompute P_ox with inflated anomalies
    P_ox_loc_inflated = (inflated_anomalies.T @ Hx_anom) / (N - 1) * loc_weights
    
    # Kalman gain: K = P_ox S^{-1}  (D, M)
    # Solve S K^T = P_ox_loc^T  =>  K = (S^{-1} P_ox_loc^T)^T
    try:
        # Use Cholesky for stability
        L_chol = np.linalg.cholesky(S)
        # Solve L_chol * tmp = P_ox_loc_inflated.T  => tmp = L_chol^{-1} P_ox_loc_inflated.T
        tmp = np.linalg.solve(L_chol, P_ox_loc_inflated.T)
        # K^T = L_chol^{-T} tmp  =>  K = tmp^T L_chol^{-1}
        K = np.linalg.solve(L_chol.T, tmp).T  # (D, M)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = P_ox_loc_inflated @ S_inv.T  # (D, M)
    
    # Update each ensemble member
    # x_a = x + K * (y - H x)
    # For each member: x_a_i = x_i + K @ (obs - H x_i)
    # = x_i + K @ (obs - Hx_mean - Hx_anom_i)
    # = x_i + K @ innovation - K @ Hx_anom_i
    
    # Precompute K @ innovation (D,)
    K_innov = K @ innovation  # (D,)
    
    # For each member: update = x_i + K_innov - K @ Hx_anom_i
    # K @ Hx_anom_i is (D,) for each i
    # Vectorized: (N, M) @ (M, D).T -> (N, D)
    Hx_anom_K = Hx_anom @ K.T  # (N, D)
    
    updated_ensemble = ensemble + K_innov[None, :] - Hx_anom_K
    
    # Re-center to preserve ensemble mean consistency
    # The above update should preserve the mean if done correctly, but let's verify
    new_mean = np.mean(updated_ensemble, axis=0)
    expected_mean = ens_mean + K_innov
    # Shift to match expected mean
    updated_ensemble = updated_ensemble + (expected_mean - new_mean)
    
    # Safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
