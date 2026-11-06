import numpy as np
import time

# EVOLVE-BLOCK-START

# Module-level state for adaptive inflation and timing
_inflation_state = {}
_start_time = None


def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LET) with:
    - Per-observation localization using Gaussian taper
    - Local EnKF analysis on patches centered on each observation
    - Adaptive multiplicative inflation with temporal memory
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)
    L = len(obs_idx)

    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result

    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))

    # Background mean
    mean_b = np.mean(X_b, axis=0)

    # Localization parameters
    loc_range = 4.0  # Gaussian localization range (ring distance)

    # Initialize analysis to background
    X_a = X_b.copy()

    # For each observed index, perform local analysis
    for j, oj in enumerate(obs_idx):
        # Determine local patch: indices within localization range
        state_idx = np.arange(N)
        diffs = np.abs(state_idx - oj)
        ring_dists = np.minimum(diffs, N - diffs)
        local_mask = ring_dists <= loc_range
        local_idx = state_idx[local_mask]
        n_local = len(local_idx)

        # Extract local background
        X_b_loc = X_b[:, local_idx]  # (M, n_local)
        mean_b_loc = np.mean(X_b_loc, axis=0)
        X_anom_loc = X_b_loc - mean_b_loc  # (M, n_local)

        # Local background covariance
        P_b_loc = (X_anom_loc.T @ X_anom_loc) / (M - 1)  # (n_local, n_local)

        # Observation: single observed variable at position oj
        # H selects the component at oj within the local patch
        oj_in_local = np.where(local_idx == oj)[0][0]
        # H is a row vector selecting oj_in_local from the local state
        # For scalar observation:
        # Innovation: y - H x_b for each member
        Hx = X_b_loc[:, oj_in_local]  # (M,)
        innovation = y_obs[:, j] - Hx  # (M,)

        # Observation covariance
        S = P_b_loc[oj_in_local, oj_in_local] + obs_var

        # Compute Kalman gain for scalar observation
        # K_i = P_b(i, oj) / S
        P_b_col = P_b_loc[:, oj_in_local]  # (n_local,)
        K = P_b_col / S  # (n_local,)

        # Local analysis update
        # X_a_loc = X_b_loc + innovation[:, None] * K[None, :]
        X_a_loc = X_b_loc + innovation[:, None] * K[None, :]

        # Store local analysis
        X_a[:, local_idx] = X_a_loc

    # Adaptive multiplicative inflation
    spread_b = np.std(X_b, axis=0, ddof=1)
    spread_a = np.std(X_a, axis=0, ddof=1)

    if N not in _inflation_state:
        _inflation_state[N] = np.ones(N)
    prev_inflation = _inflation_state[N]

    min_ratio = 0.75
    max_inflation = 1.20
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-8:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                target = min(max_inflation, min_ratio / ratio)
                inflation_factors[i] = 0.4 * prev_inflation[i] + 0.6 * target
            else:
                inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0
        else:
            inflation_factors[i] = 0.7 * prev_inflation[i] + 0.3 * 1.0

    _inflation_state[N] = inflation_factors.copy()

    if np.any(inflation_factors > 1.0):
        mean_a = np.mean(X_a, axis=0)
        X_a = mean_a[None, :] + inflation_factors[None, :] * (X_a - mean_a[None, :])

    # Safety: ensure finite values and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)

    if not np.all(np.isfinite(X_a)):
        X_a = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        X_a[:, obs_idx] += 0.5 * (obs_arr - X_a[:, obs_idx])
        np.clip(X_a, -100.0, 100.0, out=X_a)

    return X_a

# EVOLVE-BLOCK-END
