"""Stochastic Ensemble Kalman Filter with LETKF-style local updates for Lorenz-96."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _circular_distance_matrix(n):
    """Compute circular distance matrix for periodic domain of size n."""
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, n - d)
    return d


def _gaspari_cohn(d, r=4.0, a=2.0):
    """Gaspari-Cohn correlation function."""
    d = np.asarray(d)
    out = np.zeros_like(d, dtype=float)
    # r is the radius, a is the taper width
    # For d <= r - a: 1
    # For r - a < d < r + a: taper
    # For d >= r + a: 0
    inner = r - a
    outer = r + a
    mask1 = d <= inner
    mask3 = d >= outer
    mask2 = ~(mask1 | mask3)
    out[mask1] = 1.0
    if np.any(mask2):
        d2 = d[mask2]
        # Gaspari-Cohn formula for the taper region
        pi_d = np.pi * d2
        pi_r = np.pi * r
        pi_a = np.pi * a
        val = (
            (5.0 + 4.0 * np.cos(pi_d / a) + 0.5 * np.cos(2.0 * pi_d / a))
            * (outer - d2) ** 5
            + 60.0 * np.sin(pi_d / a) * (outer - d2) ** 4
            - 20.0 * np.cos(pi_d / a) * (outer - d2) ** 3
            + 20.0 * np.sin(pi_d / a) * (outer - d2) ** 2
            - 1.025 * (outer - d2)
        ) / (120.0 * a)
        out[mask2] = val
    out[mask3] = 0.0
    return out


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic EnKF with local (LETKF-style) updates and adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    deadline = 115.0  # safety margin before 120s limit

    nm, nvars = ensemble.shape  # 16, 20
    nobs = len(observed_indices)
    obs_var = float(observation_variance)

    # Precompute circular distance
    dist_mat = _circular_distance_matrix(nvars)

    # Gaspari-Cohn localization weights
    loc_weights = _gaspari_cohn(dist_mat, r=3.5, a=2.0)

    # Observation operator
    H = np.zeros((nobs, nvars))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0

    # --- Local Sequential Update (LETKF-style) ---
    # Process each observation sequentially, updating only nearby variables
    result = ensemble.copy()
    rng = np.random.default_rng()

    for k in range(nobs):
        obs_val = observations[k]
        obs_idx = int(observed_indices[k])

        # Local neighborhood: variables within distance 4 of obs_idx
        local_mask = dist_mat[obs_idx] <= 4
        local_indices = np.where(local_mask)[0]
        n_local = len(local_indices)

        if n_local == 0:
            continue

        # Extract local ensemble
        x_local = result[:, local_indices]  # (nm, n_local)
        x_mean = np.mean(x_local, axis=0)

        # Innovation
        innov = obs_val - x_mean[int(obs_idx) - local_indices[0]] if obs_idx in local_indices else obs_val - x_mean[0]
        # More directly: the observation is of variable obs_idx, which is in local_indices
        # Find position of obs_idx in local_indices
        pos_in_local = np.where(local_indices == obs_idx)[0]
        if len(pos_in_local) == 0:
            continue
        pos_in_local = pos_in_local[0]
        innov = obs_val - x_mean[pos_in_local]

        # Local ensemble covariance (sample)
        anomalies_local = x_local - x_mean
        # Local covariance: (n_local, n_local)
        C_local = (anomalies_local.T @ anomalies_local) / (nm - 1)

        # Apply localization to local covariance (further taper within local window)
        # The local indices are already within distance 4, so apply a mild additional taper
        local_dist = dist_mat[np.ix_(local_indices, local_indices)]
        local_loc = _gaspari_cohn(local_dist, r=3.0, a=1.5)
        C_local_loc = C_local * local_loc

        # Observation variance for this single observation
        R_k = obs_var

        # Kalman gain for this local update
        # P_oo = C_local_loc[pos_in_local, pos_in_local] + R_k
        P_oo = C_local_loc[pos_in_local, pos_in_local] + R_k

        # Gain vector: K = C_local_loc[:, pos_in_local] / P_oo
        K_vec = C_local_loc[:, pos_in_local] / P_oo  # (n_local,)

        # Stochastic update: add noise to observation
        # y_noisy = obs_val + N(0, R_k) -- but we already have noisy obs
        # For stochastic EnKF, we inject noise into the update
        # x_a = x_f + K * (y - H x_f) + sqrt(R) * K * eps  where eps ~ N(0,1)
        # This is equivalent to: x_a = x_f + K * (y_noisy - H x_f)
        # where y_noisy = y + sqrt(R)*eps

        # Deterministic part
        update_det = K_vec * innov  # (n_local,)

        # Stochastic part: add noise
        # The stochastic term is K * sqrt(R) * eps for each member
        # But in the standard stochastic EnKF, we add to each member:
        # x_a^m = x_f^m + K * (y - H x_f^m)  (this is the "perturbed observation" approach)
        # Or equivalently: x_a^m = x_f^m + K * (y - H x_mean) + K * (H x_f^m - H x_mean) + K * sqrt(R) * eps^m
        # The second term is the "perturbation" part

        # Use the perturbed observation approach:
        # For each member, compute innovation with that member's state
        eps = rng.standard_normal(nm)
        noise_term = K_vec * np.sqrt(R_k) * eps  # (nm, n_local)

        # Update each member
        for m in range(nm):
            # Innovation for member m
            innov_m = obs_val - x_local[m, pos_in_local]
            result[m, local_indices] = x_local[m] + K_vec * innov_m + noise_term[m]

    # --- Adaptive Inflation ---
    # Check ensemble spread vs expected
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean
    ens_cov_diag = np.var(result, axis=0, ddof=1)

    # Expected variance from observations: for observed variables, the spread should be ~ obs_var + model uncertainty
    # For simplicity, use a gentle inflation based on the ratio of innovation variance to expected
    # Compute innovation variance for observed variables
    innov_all = observations - result[np.arange(nm)[:, None], observed_indices][np.arange(nm), np.arange(nobs)]
    # Actually: for each member m and observation k, innovation = y_k - x_m[obs_idx_k]
    innov_matrix = observations[None, :] - result[:, observed_indices]  # (nm, nobs)
    innov_var = np.var(innov_matrix, axis=0)  # (nobs,)
    mean_innov_var = np.mean(innov_var)

    # Expected innovation variance ~ obs_var + average ensemble variance at observed locations
    ens_var_at_obs = np.mean(ens_cov_diag[observed_indices])
    expected_innov_var = obs_var + ens_var_at_obs

    if expected_innov_var > 1e-10:
        ratio = mean_innov_var / expected_innov_var
        # Inflation: if ratio < 1, ensemble is under-dispersed
        inflation = 1.0 + 0.15 * max(0.0, 1.0 - ratio)
        inflation = min(inflation, 1.15)
    else:
        inflation = 1.0

    # Apply inflation
    if inflation > 1.0:
        result = ens_mean + anomalies * np.sqrt(inflation)

    # --- Safety ---
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)

    if time.time() - t0 > deadline:
        pass

    return result

# EVOLVE-BLOCK-END
