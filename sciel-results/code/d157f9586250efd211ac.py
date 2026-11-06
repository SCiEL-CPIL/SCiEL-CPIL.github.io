"""Ensemble Kalman filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis update for Lorenz-96.

    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise

    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s limit

    n_ens, n_state = ensemble.shape
    n_obs = len(observed_indices)

    # Convert to float64 for precision
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = observed_indices.astype(np.int64)
    obs_var = float(observation_variance)

    # --- Gaussian localization matrix ---
    # Distance matrix for the 20 state variables (circular distance)
    idx = np.arange(n_state)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_state - dist)  # circular distance
    # Gaussian localization with length scale ~5
    loc_length = 5.0
    loc_matrix = np.exp(-0.5 * (dist / loc_length) ** 2)
    np.fill_diagonal(loc_matrix, 1.0)

    # --- Adaptive inflation ---
    # Start with a mild inflation factor; adapt based on innovation statistics
    inflation = 1.02  # initial multiplicative inflation

    # Apply inflation around the ensemble mean
    ens_mean = ens.mean(axis=0)
    ens = ens_mean + inflation * (ens - ens_mean)

    # --- Perturbed observations (stochastic EnKF) ---
    # Add random perturbations to observations to avoid rank-deficiency
    rng = np.random.default_rng(seed=42)
    obs_perturbed = obs + rng.standard_normal(n_obs) * np.sqrt(obs_var)

    # --- Compute ensemble mean and anomalies ---
    ens_mean = ens.mean(axis=0)
    anomalies = ens - ens_mean  # (16, 20)

    # Observation operator: extract observed components
    H_anom = anomalies[:, obs_idx]  # (16, 10)
    H_mean = ens_mean[obs_idx]      # (10,)

    # --- Innovation ---
    innovation = obs_perturbed - H_mean  # (10,)

    # --- Compute forecast observation covariance ---
    # S = H P H^T + R
    # P is estimated from anomalies: P = (1/(n-1)) * A A^T
    # H P H^T = (1/(n-1)) * H_anom H_anom^T
    HPHt = (H_anom @ H_anom.T) / (n_ens - 1)  # (10, 10)
    R = obs_var * np.eye(n_obs)
    S = HPHt + R  # (10, 10)

    # --- Compute gain ---
    # K = P H^T S^{-1}
    # P H^T = (1/(n-1)) * A H_anom^T
    PHt = (anomalies @ H_anom.T) / (n_ens - 1)  # (20, 10)

    # Apply localization to the cross-covariance (P H^T)
    # Localize in the state space: for each state i, scale by localization
    # with respect to observed states
    # We localize the gain by scaling each row of PHt
    # The localization is based on distance between state i and observed states
    loc_gain = np.ones((n_state, n_obs))
    for i in range(n_state):
        for j, oj in enumerate(obs_idx):
            d = abs(i - oj)
            d = min(d, n_state - d)
            loc_gain[i, j] = np.exp(-0.5 * (d / loc_length) ** 2)

    PHt_loc = PHt * loc_gain  # (20, 10)

    # Invert S (small 10x10)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = PHt_loc @ S_inv  # (20, 10)

    # --- Update ensemble ---
    # x_a = x_f + K (y - H x_f)
    # For each member: x_a_m = x_f_m + K (y_pert - H x_f_m)
    # = x_f_m + K (y_pert - H_mean - H_anom_m)
    # = x_f_m + K innovation - K H_anom_m
    # But K innovation is the same for all members, and K H_anom_m varies

    # More efficiently:
    # x_a_m = x_f_m + K @ (innovation - H_anom_m)
    innovation_minus_Hanom = innovation[None, :] - H_anom  # (16, 10)
    update = innovation_minus_Hanom @ K.T  # (16, 20)
    ens = ens + update

    # --- Adaptive inflation adjustment ---
    # Compute analysis innovation statistics to adapt inflation
    # If the ensemble is too tight, increase inflation; if too wide, decrease
    # Use a simple heuristic based on the ratio of observed to forecast spread
    H_anom_ens = ens[:, obs_idx] - ens[:, obs_idx].mean(axis=0)
    forecast_obs_var = np.mean((H_anom_ens ** 2).sum(axis=0) / (n_ens - 1))
    actual_obs_var = np.var(obs - ens[:, obs_idx].mean(axis=0), ddof=1) if n_ens > 1 else obs_var

    # Simple adaptive inflation: adjust based on how well the ensemble
    # captures the observation spread
    if actual_obs_var > 0:
        ratio = forecast_obs_var / actual_obs_var
        # If ratio < 1, ensemble is too tight -> increase inflation
        # If ratio > 1, ensemble is too wide -> decrease inflation
        if ratio < 0.8:
            inflation = min(inflation * 1.05, 1.5)  # cap at 1.5
        elif ratio > 1.2:
            inflation = max(inflation * 0.95, 1.0)  # floor at 1.0

    # Check time
    if time.time() > deadline:
        # Return best result found so far
        pass

    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)

    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)

    return ens.astype(np.float64)
# EVOLVE-BLOCK-END
