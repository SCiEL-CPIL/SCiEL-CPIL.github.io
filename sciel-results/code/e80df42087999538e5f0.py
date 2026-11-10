"""Lorenz-96 ensemble assimilation using Localized EnKF with adaptive inflation."""
import numpy as np
import time


def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function (vectorized)."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    r = d / c
    m1 = r <= 0.5
    m2 = (r > 0.5) & (r <= 1.5)
    m3 = (r > 1.5) & (r <= 2.5)

    r1 = r[m1]
    result[m1] = (-5.0 / 4.0) * r1**5 + 5.0 * r1**4 - 10.0 * r1**3 + 5.0 * r1

    r2 = r[m2]
    result[m2] = (1.0 / 12.0) * r2**5 - (5.0 / 4.0) * r2**4 + (5.0 / 2.0) * r2**3 - 5.0 * r2**2 + (25.0 / 12.0) * r2 + 0.5

    r3 = r[m3]
    result[m3] = (-1.0 / 12.0) * r3**5 + (5.0 / 4.0) * r3**4 - (5.0 / 2.0) * r3**3 + 5.0 * r3**2 - (25.0 / 12.0) * r3 + 35.0 / 12.0

    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter (EnKF) with Gaspari-Cohn localization
    and adaptive multiplicative inflation for Lorenz-96 assimilation.

    Parameters
    ----------
    ensemble : (16, 20) array
        Forecast ensemble members.
    observations : (10,) or (16, 10) array
        Noisy observations of even-indexed coordinates.
    observed_indices : array-like of int
        Indices of observed coordinates.
    observation_variance : float
        Observation noise variance.

    Returns
    -------
    updated_ensemble : (16, 20) array
        Analysis ensemble.
    """
    t0 = time.time()
    deadline = t0 + 110.0

    ens = np.asarray(ensemble, dtype=np.float64).copy()
    M, N = ens.shape
    obs_idx = np.asarray(observed_indices, dtype=int)
    K = len(obs_idx)

    # Observation handling: support both (K,) and (M, K)
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 1:
        obs_vec = obs_arr
        obs_matrix = np.tile(obs_vec, (M, 1))
    else:
        obs_vec = obs_arr.mean(axis=0)
        obs_matrix = obs_arr

    # Ensemble mean and anomalies
    ens_mean = ens.mean(axis=0)
    anomalies = ens - ens_mean  # (M, N)

    # Full background covariance
    P = (anomalies.T @ anomalies) / (M - 1)  # (N, N)

    # Gaspari-Cohn localization on the ring
    corr_len = 3.5
    indices = np.arange(N)
    dist = np.abs(indices[:, None] - indices[None, :])
    dist = np.minimum(dist, N - dist)
    loc = _gaspari_cohn(dist, corr_len)

    # Localized covariance
    P_loc = P * loc

    # Observation operator H: (K, N)
    H = np.zeros((K, N))
    H[np.arange(K), obs_idx] = 1.0

    # Observation covariance
    R = observation_variance * np.eye(K)

    # Innovation (mean)
    innovation = obs_vec - H @ ens_mean  # (K,)

    # Innovation covariance
    S = H @ P_loc @ H.T + R  # (K, K)

    # Solve for gain application: gain @ innovation = P_loc @ H.T @ S^{-1} @ innovation
    try:
        S_inv_innov = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        S_inv_innov = S_inv @ innovation

    update_mean = P_loc @ H.T @ S_inv_innov  # (N,)

    # Updated mean
    mean_a = ens_mean + update_mean

    # Compute gain matrix for member updates
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    gain = P_loc @ H.T @ S_inv  # (N, K)

    # Perturbed observations update
    innovations_m = obs_matrix - ens[:, obs_idx]  # (M, K)
    ens_a = ens + innovations_m @ gain.T  # (M, N)

    # Adaptive multiplicative inflation
    # Compare ensemble spread at observed points to observation error
    ens_spread_obs = np.var(ens_a[:, obs_idx], axis=0)  # (K,)
    obs_var = observation_variance

    if obs_var > 0:
        spread_ratio = np.mean(ens_spread_obs) / obs_var
    else:
        spread_ratio = 1.0

    # Target: slight over-dispersion to avoid collapse
    target = 1.10
    deviation = (spread_ratio - target) / target
    beta = 0.25
    inflation = 1.0 + beta * np.tanh(-deviation)
    inflation = np.clip(inflation, 0.95, 1.40)

    # Apply inflation around the analysis mean
    ens_a = mean_a + inflation * (ens_a - mean_a)

    # Safety: ensure finite and bounded
    ens_a = np.nan_to_num(ens_a, nan=0.0, posinf=100.0, neginf=-100.0)
    ens_a = np.clip(ens_a, -100.0, 100.0)

    # Deadline guard
    if time.time() > deadline:
        return ens_a

    return ens_a
