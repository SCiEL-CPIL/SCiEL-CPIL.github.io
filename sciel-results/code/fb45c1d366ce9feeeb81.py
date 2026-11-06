"""Ensemble Kalman filter for Lorenz-96 partial-observation assimilation."""
import numpy as np

# EVOLVE-BLOCK-START

def _ring_distance(i, j, n):
    """Minimum ring distance between indices i and j on a ring of length n."""
    d = abs(i - j)
    return min(d, n - d)


def _localize_matrix(n, decay):
    """Exponential localization matrix on a ring of length n."""
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n - dist)
    return np.exp(-decay * dist)


def analyze(ensemble, observations, observed_indices, observation_variance):
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)

    n_members, n_state = ensemble.shape
    result = ensemble.copy()

    # Guard against degenerate inputs.
    if n_members < 2 or n_state < 3 or len(observed_indices) == 0:
        # Fall back to gentle nudging on observed coordinates.
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        return result

    # --- Localization parameters -------------------------------------------
    # Ring of length n_state; decay controls correlation length.
    decay = 1.2 / max(n_state - 1, 1)
    loc_bg = _localize_matrix(n_state, decay)          # background localization
    loc_obs = _localize_matrix(n_state, decay)         # observation localization

    # --- Ensemble statistics ------------------------------------------------
    mean = ensemble.mean(axis=0)
    # Centered anomalies (n_members x n_state)
    X = ensemble - mean

    # Background covariance (localized): P = X^T X / (m-1), then localize.
    # P[i,j] = sum_k X[k,i] X[k,j] / (m-1) * loc_bg[i,j]
    XtX = X.T @ X / (n_members - 1.0)
    P = XtX * loc_bg

    # Observation noise covariance (diagonal, localized by distance from obs).
    # For each observed coordinate, noise variance = observation_variance.
    obs_var = float(observation_variance)

    # --- Localized ensemble Kalman update ----------------------------------
    # We update each state coordinate i by projecting the anomaly onto the
    # observation subspace, weighted by the localized background covariance
    # and the (localized) observation noise.
    #
    # Standard EKF: x_a = x_b + P_b H^T (H P_b H^T + R)^-1 (y - H x_b)
    # Here H selects observed_indices. We do a per-member (deterministic)
    # form using the ensemble directly to avoid explicit P inversion issues:
    #
    #   K = P_b H^T (H P_b H^T + R)^-1
    #   x_a = x_b + K (y - H x_b)
    #
    # We compute K (n_state x n_obs) and apply to the innovation.

    n_obs = len(observed_indices)

    # H P_b H^T : n_obs x n_obs
    HPbHt = P[np.ix_(observed_indices, observed_indices)]
    # R : diagonal with localized observation noise
    # For each observed coordinate, effective noise = obs_var * loc_obs[i,i] (=1)
    # but we localize the cross-noise to keep the system well-conditioned.
    R = np.diag(np.full(n_obs, obs_var))
    # Add a small regularization for numerical stability.
    S = HPbHt + R + 1e-8 * np.eye(n_obs)

    # Innovation: y - H mean
    innovation = observations - mean[observed_indices]

    # Gain: K = P_b H^T S^-1   (n_state x n_obs)
    PbHt = P[:, observed_indices]
    try:
        # Solve S^T K^T = PbHt^T  ->  K = (S^-1 PbHt)^T
        K = np.linalg.solve(S, PbHt.T).T
    except np.linalg.LinAlgError:
        K = np.linalg.lstsq(S, PbHt.T, rcond=None)[0].T

    # Update the mean:
    mean_a = mean + K @ innovation

    # Update each member (deterministic EKF, preserving spread via inflation):
    # x_a_m = x_b_m + K (y - H x_b_m)
    #        = x_b_m + K (innovation - H (x_b_m - mean))
    Xobs = X[:, observed_indices]            # n_members x n_obs
    member_innovation = innovation[None, :] - Xobs
    result = ensemble + member_innovation @ K.T

    # --- Inflation about the analysis mean ---------------------------------
    # Adaptive-ish inflation: measure spread of analysis vs. innovation,
    # inflate to keep the ensemble from collapsing.
    # Target: keep the ratio of ensemble spread to observation noise reasonable.
    # Compute a simple spread ratio on observed coordinates.
    spread_after = np.std(result[:, observed_indices], axis=0)
    # Expected spread after a perfect update ~ sqrt(P_obs) ; we inflate if too small.
    # Use a conservative, clamped inflation factor.
    # Estimate background spread on observed coords.
    spread_bg = np.std(ensemble[:, observed_indices], axis=0)
    # If analysis spread collapsed relative to background, inflate.
    ratio = np.mean(spread_after / (spread_bg + 1e-8))
    # Inflation factor: gentle, in [1.0, 1.5].
    if ratio < 0.7:
        infl = 1.0 + 0.5 * (0.7 - ratio) / 0.7
        infl = min(infl, 1.5)
    else:
        infl = 1.0
    result = mean_a + infl * (result - mean_a)

    # --- Safety: clamp to finite and bounded --------------------------------
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)

    return result

# EVOLVE-BLOCK-END
