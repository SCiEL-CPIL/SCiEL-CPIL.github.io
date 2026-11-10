import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter (LETKF) with adaptive inflation.
    Uses a local analysis window around each state variable to avoid spurious
    long-range correlations.
    """
    start_time = time.time()

    M, N = ensemble.shape
    K = len(observed_indices)

    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)

    # Precompute which observed indices are within a window of each state
    window = 4  # window size: i-window .. i+window (mod N)

    # Adaptive inflation: compute spread at observed locations
    ens_spread = np.var(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    obs_var = observation_variance

    # Simple adaptive inflation based on spread/observation variance ratio
    ratio = mean_spread / obs_var if obs_var > 0 else 1.0
    # Target ratio ~ 1.2
    target = 1.2
    if ratio < 0.7:
        inflation = 1.0 + 0.15 * (0.7 - ratio) / 0.7
    elif ratio > 2.0:
        inflation = 1.0 - 0.05 * (ratio - 2.0) / 2.0
    else:
        inflation = 1.0 + 0.02 * (target - ratio) / target
    inflation = float(np.clip(inflation, 0.95, 1.30))

    # Apply inflation around the ensemble mean
    mean = np.mean(ensemble, axis=0)
    ensemble = mean + inflation * (ensemble - mean)

    # Local analysis: for each state variable i, use only nearby observations
    result = np.zeros_like(ensemble)

    for i in range(N):
        # Determine which observed indices are within the window
        local_obs = []
        for k, idx in enumerate(observed_indices):
            d = abs(i - idx)
            d = min(d, N - d)
            if d <= window:
                local_obs.append(k)

        if len(local_obs) == 0:
            result[:, i] = ensemble[:, i]
            continue

        local_obs = np.array(local_obs)
        K_local = len(local_obs)

        # Local observation operator: selects local_obs from the state
        # Local state: just variable i (1-dimensional)
        # We need to do a 1D transform Kalman filter update

        # Ensemble mean and anomalies for variable i
        x_mean = np.mean(ensemble[:, i])
        x_anom = ensemble[:, i] - x_mean  # (M,)

        # Local observations
        y = observations[:, local_obs]  # (M, K_local)
        y_mean = np.mean(y, axis=0)  # (K_local,)

        # Sample covariance between x_anom and y_anom
        y_anom = y - y_mean  # (M, K_local)
        # C_xy = x_anom @ y_anom.T / (M-1)  -> (1, K_local)
        # C_yy = y_anom.T @ y_anom / (M-1)  -> (K_local, K_local)

        # For numerical stability with small M, add small regularization
        C_xy = (x_anom[:, np.newaxis] @ y_anom) / (M - 1)  # (1, K_local)
        C_yy = (y_anom.T @ y_anom) / (M - 1)  # (K_local, K_local)

        R_local = observation_variance * np.eye(K_local)
        S = C_yy + R_local  # (K_local, K_local)

        # Innovation
        innovation = y_mean - x_mean  # (K_local,)

        # Gain: C_xy @ S^{-1}  -> (1, K_local)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        gain = C_xy @ S_inv  # (1, K_local)
        gain_i = gain[0]  # (K_local,)

        # Update mean
        x_a_mean = x_mean + np.dot(gain_i, innovation)

        # Transform Kalman filter for the anomalies
        # We need to compute the transform T such that T @ x_anom gives updated anomalies
        # For 1D state, this is just scaling

        # Compute the analysis variance
        var_x = np.var(ensemble[:, i], ddof=1)
        # Analysis variance = var_x - C_xy @ S^{-1} @ C_xy.T
        # C_xy @ S^{-1} @ C_xy.T is scalar
        corr_term = np.dot(gain_i, C_xy[0])
        var_x_a = var_x - corr_term
        if var_x_a < 1e-12:
            var_x_a = 1e-12

        # Scaling factor for anomalies
        scale = np.sqrt(var_x_a / var_x) if var_x > 1e-12 else 1.0

        # Updated anomalies
        x_anom_a = scale * x_anom

        # Final updated ensemble for variable i
        result[:, i] = x_a_mean + x_anom_a

    # Safety: finite and bounded
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100, 100)

    # Deadline guard
    if time.time() - start_time > 140:
        return result

    return result
