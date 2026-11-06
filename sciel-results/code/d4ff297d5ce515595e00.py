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


def _gaspari_cohn(d, L):
    """Gaspari-Cohn localization function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=np.float64)
    mask1 = d <= L
    mask2 = (d > L) & (d <= 2.0 * L)
    mask3 = d > 2.0 * L
    L2 = L * L
    result[mask1] = (-1.0 / (2.0 * L2) * d[mask1]**5 +
                     5.0 / (4.0 * L2) * d[mask1]**4 -
                     5.0 / (12.0 * L2) * d[mask1]**3 +
                     1.0)
    result[mask2] = (1.0 / (2.0 * L2) * d[mask2]**5 -
                     5.0 / (4.0 * L2) * d[mask2]**4 +
                     5.0 / (12.0 * L2) * d[mask2]**3 -
                     5.0 / (2.0 * L) * d[mask2] +
                     5.0 / 3.0)
    result[mask3] = 0.0
    return result


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with Gaspari-Cohn localization, ensemble adjustment,
    and multiplicative adaptive inflation.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)

    # P_obs = cov(observed) + R
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)

    # Cross-covariance P_xH^T
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)

    # Gaspari-Cohn localization: minimum-image distance on ring
    d_idx = np.arange(D)
    L = 3.0  # localization length scale
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = _gaspari_cohn(dist, L)

    PHt_loc = PHt * loc_weights  # (D, M)

    # Also localize the observation covariance for stability
    S_loc = S.copy()
    # For diagonal R, localization mainly affects P_obs off-diagonals
    # Keep S as is but use localized cross-cov

    # Compute S_inv
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    # Ensemble adjustment form (Jazwinski):
    # delta_x_i = anomalies_i @ (S_inv @ PHt_loc.T) @ (obs_vec - ens_obs_i) / (N-1)
    # More precisely: gain = PHt_loc @ S_inv  (D, M)
    # delta_x_i = gain @ (obs_vec - ens_obs_i)
    # But for ensemble adjustment:
    # delta_x_i = (1/(N-1)) * anomalies_i @ S_inv @ PHt_loc.T @ (obs_vec - ens_obs_i) ... no
    # Standard EKF: delta_x_i = K @ (y - H x_i) where K = P_xH^T S^{-1}
    # P_xH^T = PHt_loc (already localized)
    # K = PHt_loc @ S_inv

    K = PHt_loc @ S_inv  # (D, M)

    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)

    delta_x = innovations @ K.T  # (N, D)

    updated_ensemble = ensemble + delta_x  # (N, D)

    # Multiplicative adaptive inflation
    # Measure spread vs observation noise at observed points
    ens_spread_obs = np.std(updated_ensemble[:, observed_indices], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    target_ratio = 1.15

    if obs_noise_std > 0 and mean_spread > 0:
        ratio = mean_spread / obs_noise_std
        # Smooth inflation adjustment
        inflation = 1.0 + 0.10 * (target_ratio - ratio) / max(ratio, 0.1)
    else:
        inflation = 1.0

    inflation = np.clip(inflation, 0.98, 1.08)

    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)

    # Safety
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)

    return updated_ensemble


def _dynamics_constrained_update(ensemble, obs_vec, observed_indices, observation_variance, F):
    """
    Use Lorenz-96 dynamics to constrain the update.
    The key insight: for Lorenz-96, the state is highly correlated along the ring.
    We can use the dynamics to propagate the ensemble forward and backward
    to get a better estimate of the unobserved states.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()

    # First, do a standard EKF update
    ekf_result = _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance)

    # Then, use dynamics to refine: propagate each member forward by a small step
    # and use the observation constraint to pull back
    # This is a simplified "dynamics-informed" correction

    # Estimate F from the ensemble (in steady state, F ≈ mean of x)
    # But F is actually given by the task as 6, 8, or 10
    # We can estimate it: in steady state, E[(x_{i+1}-x_{i-2})x_{i-1}] ≈ E[x_i] - F
    # So F ≈ mean(x) - mean((x_{i+1}-x_{i-2})x_{i-1})
    ens_mean = np.mean(ekf_result, axis=0)
    rhs_mean = np.mean(_lorenz96_rhs(ekf_result, 0.0), axis=0)
    # F ≈ -mean(rhs) + mean(x) ... actually rhs = (x_{i+1}-x_{i-2})x_{i-1} - x + F
    # So mean(rhs) = mean((x_{i+1}-x_{i-2})x_{i-1}) - mean(x) + F
    # F = mean(rhs) - mean((x_{i+1}-x_{i-2})x_{i-1}) + mean(x)
    # This is circular. Let's just use the EKF result directly.

    return ekf_result


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
    deadline = start_time + 115.0  # safety margin

    N, D = ensemble.shape
    M = len(observed_indices)

    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()

    # Primary: EKF with Gaspari-Cohn localization and adaptive inflation
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance
    )

    # Deadline guard
    if time.time() > deadline:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result

    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)

    return result

# EVOLVE-BLOCK-END
