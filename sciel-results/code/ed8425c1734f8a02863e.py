"""Ensemble analysis for partial-observation Lorenz-96 assimilation.

This implementation uses a Localized Ensemble Kalman Filter (LEKF) with:
  - Gaussian localization for cross-covariances
  - Adaptive multiplicative inflation based on innovation spread
  - A deadline guard to stay within the 150 s wall-clock budget
  - Safety clipping to keep outputs finite and bounded
"""

import numpy as np
import time

# EVOLVE-BLOCK-START

# ── Global deadline tracking ────────────────────────────────────────────
_start_time = None
_DEADLINE_FRACTION = 0.88  # Use 88 % of the 150 s budget


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


def _periodic_dist(d, obs_i, D):
    """Periodic (circular) distance between state index d and observation index obs_i."""
    dist = abs(d - obs_i)
    return min(dist, D - dist)


def _compute_localization_matrix(observed_indices, D, L_loc):
    """Pre-compute the (D, M) localization weight matrix."""
    M = len(observed_indices)
    W = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = _periodic_dist(d, obs_i, D)
            W[d, m_idx] = _gaussian_localization(dist, L_loc)
    return W


def _adaptive_inflation(innovation, S_diag, base_inflation=1.0):
    """Compute multiplicative inflation factor from innovation statistics.

    Uses a smooth, continuous mapping from the innovation-to-expected ratio
    to an inflation multiplier in [0.7, 1.8].
    """
    innov_std = np.std(innovation)
    expected_innov_std = np.sqrt(np.mean(S_diag))

    if expected_innov_std < 1e-12:
        return base_inflation

    ratio = innov_std / expected_innov_std

    # Smooth S-curve mapping: ratio=1 → 1.0, ratio>1 → inflate, ratio<1 → deflate
    # tanh-based: gentle at extremes, responsive in the mid-range
    inflation = base_inflation + 0.45 * np.tanh(1.2 * (ratio - 1.0))
    return np.clip(inflation, 0.7, 1.8)


def _ekf_gain_update(ensemble, obs_vec, observed_indices, observation_variance,
                     L_loc=2.0, inflation_boost=1.0):
    """Core LEKF update: compute gain and apply to ensemble members.

    Parameters
    ----------
    ensemble : (N, D)
    obs_vec : (M,)
    observed_indices : (M,)
    observation_variance : float
    L_loc : localization length scale
    inflation_boost : additional multiplier applied to inflation factor

    Returns
    -------
    updated_ensemble : (N, D)
    """
    N, D = ensemble.shape
    M = len(observed_indices)

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean

    Hx = ensemble[:, observed_indices]
    Hx_mean = np.mean(Hx, axis=0)
    Hx_anom = Hx - Hx_mean

    innovation = obs_vec - Hx_mean

    P_oo = (Hx_anom.T @ Hx_anom) / (N - 1)
    P_ox = (anomalies.T @ Hx_anom) / (N - 1)

    R = observation_variance * np.eye(M)
    S = P_oo + R

    # Localization
    loc_W = _compute_localization_matrix(observed_indices, D, L_loc)
    P_ox_loc = P_ox * loc_W

    # Adaptive inflation
    S_diag = np.diag(S)
    infl_factor = _adaptive_inflation(innovation, S_diag) * inflation_boost

    inflated_anoms = anomalies * infl_factor
    P_ox_loc_inf = (inflated_anoms.T @ Hx_anom) / (N - 1) * loc_W

    # Kalman gain via Cholesky
    try:
        Lc = np.linalg.cholesky(S + 1e-12 * np.eye(M))
        tmp = np.linalg.solve(Lc, P_ox_loc_inf.T)
        K = np.linalg.solve(Lc.T, tmp).T
    except np.linalg.LinAlgError:
        S_reg = S + 1e-8 * np.eye(M)
        K = P_ox_loc_inf @ np.linalg.pinv(S_reg)

    K_innov = K @ innovation
    Hx_anom_K = Hx_anom @ K.T

    updated = ensemble + K_innov[None, :] - Hx_anom_K

    # Re-center
    new_mean = np.mean(updated, axis=0)
    expected_mean = ens_mean + K_innov
    updated = updated + (expected_mean - new_mean)

    return updated


def _particle_weighted_update(ensemble, obs_vec, observed_indices,
                              observation_variance, N_particles=200,
                              bandwidth_factor=1.5):
    """Nonlinear particle-filter-style update using importance weights
    and a deterministic kernel-density approximation of the posterior.

    With only 16 ensemble members in 20-D state space, a full particle filter
    is infeasible.  Instead we:
      1. Compute Gaussian likelihood weights for each member given the observations.
      2. Use weighted mean and a covariance-weighted spread to form the posterior.
      3. Blend with the linear LEKF result for stability.

    Parameters
    ----------
    ensemble : (N, D)
    obs_vec : (M,)
    observed_indices : (M,)
    observation_variance : float
    N_particles : int (unused – kept for interface clarity)
    bandwidth_factor : float – widens the KDE bandwidth relative to ensemble spread

    Returns
    -------
    updated_ensemble : (N, D)
    """
    N, D = ensemble.shape
    M = len(observed_indices)

    ens_mean = np.mean(ensemble, axis=0)
    ens_std = np.std(ensemble, axis=0) + 1e-10  # (D,)

    Hx = ensemble[:, observed_indices]
    Hx_mean = np.mean(Hx, axis=0)

    # Gaussian log-likelihood for each member
    diff = Hx - obs_vec  # (N, M)
    log_w = -0.5 * np.sum(diff ** 2, axis=1) / observation_variance  # (N,)

    # Stabilise weights
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w_sum = np.sum(w)
    if w_sum < 1e-15:
        w = np.ones(N) / N
    else:
        w = w / w_sum

    # Weighted posterior mean
    post_mean = np.sum(w[:, None] * ensemble, axis=0)  # (D,)

    # Weighted anomalies
    post_anoms = ensemble - post_mean[None, :]

    # Weighted covariance (for spread estimation)
    # Use the weighted second moment
    w2 = w[:, None] ** 2  # (N,1)
    post_cov = (w2 * post_anoms ** 2).sum(axis=0)  # (D,) diagonal approx
    post_std = np.sqrt(np.maximum(post_cov, 1e-12))

    # Rebuild ensemble around the posterior mean with spread
    # Use the original anomaly directions scaled to match posterior spread
    orig_std = np.std(ensemble, axis=0) + 1e-10
    scale = post_std / orig_std  # (D,)

    updated = post_mean[None, :] + (ensemble - ens_mean[None, :]) * scale[None, :]

    return updated


def _blend_updates(ekf_result, pf_result, alpha=0.7):
    """Blend LEKF and particle-filter results.

    alpha is the weight on the LEKF result; (1-alpha) on the PF result.
    """
    return alpha * ekf_result + (1.0 - alpha) * pf_result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update combining a Localized EnKF with a
    particle-weighted nonlinear correction.

    Parameters
    -----------
    ensemble : (16, 20) array – ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array – noisy obs of even-indexed coords
    observed_indices : array of 10 indices – even indices [0,2,4,...,18]
    observation_variance : float – variance of observation noise (0.49)

    Returns
    --------
    updated_ensemble : (16, 20) array – updated ensemble
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    if elapsed > 150 * _DEADLINE_FRACTION:
        return np.clip(ensemble, -100.0, 100.0)

    N, D = ensemble.shape
    M = len(observed_indices)

    obs_arr = np.asarray(observations, dtype=float)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.astype(float, copy=False)

    # ── Path 1: Localized Ensemble Kalman Filter ────────────────────────
    L_loc_ekf = 2.0  # tighter than the seed's 2.5
    ekf_result = _ekf_gain_update(
        ensemble, obs_vec, observed_indices,
        observation_variance, L_loc=L_loc_ekf, inflation_boost=1.0
    )

    # ── Path 2: Particle-weighted nonlinear correction ─────────────────
    pf_result = _particle_weighted_update(
        ensemble, obs_vec, observed_indices,
        observation_variance, bandwidth_factor=1.5
    )

    # ── Blend ──────────────────────────────────────────────────────────
    # Give more weight to the LEKF (well-tuned, linear-optimal) but
    # let the PF correct for non-Gaussian tails.
    alpha = 0.72
    updated = _blend_updates(ekf_result, pf_result, alpha=alpha)

    # ── Optional second-pass LEKF refinement on the blended ensemble ───
    # A lightweight second pass can sharpen the result without much cost.
    # Only do it if we still have time budget.
    elapsed2 = time.time() - _start_time
    if elapsed2 < 150 * _DEADLINE_FRACTION * 0.5:
        ekf2 = _ekf_gain_update(
            updated, obs_vec, observed_indices,
            observation_variance, L_loc=1.8, inflation_boost=1.0
        )
        updated = 0.6 * ekf2 + 0.4 * updated

    # ── Safety ─────────────────────────────────────────────────────────
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)

    return updated

# EVOLVE-BLOCK-END
