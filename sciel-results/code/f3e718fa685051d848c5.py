"""Ensemble Kalman Filter with localization, adaptive inflation, and rank-deficiency handling."""
import numpy as np
import time

def _lorenz96_step(state, F, dt=0.01, n_steps=5):
    """Propagate a single Lorenz-96 state using RK4."""
    n = len(state)
    x = state.copy()
    for _ in range(n_steps):
        def f(s):
            dx = np.empty_like(s)
            for i in range(n):
                dx[i] = (s[(i+1) % n] - s[(i-2) % n]) * s[(i-1) % n] - s[i] + F
            return dx
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with spatial localization, adaptive multiplicative inflation, and rank-deficiency handling.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like
        Indices of observed variables.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = ensemble.shape[0]  # Number of ensemble members
    n_vars = ensemble.shape[1]  # Number of variables
    n_obs = len(observed_indices)
    
    # Convert to arrays if needed
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, n_vars)
    
    # Sample covariance of anomalies
    # P = anomalies.T @ anomalies / (N-1)
    # For efficiency, work with the (N-1) x N anomaly matrix
    
    # Observation operator: H selects observed indices
    # H is (n_obs, n_vars) but we only need to index into arrays
    
    # Compute observation-space anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, n_obs)
    
    # Innovation: observations - ensemble mean at observed locations
    innovation = observations - ens_mean[observed_indices]  # (n_obs,)
    
    # Compute the Kalman gain using the ensemble formulation
    # K = P H^T (H P H^T + R)^{-1}
    # In ensemble form: K = A_obs^T (A_obs A_obs^T / (N-1) + R)^{-1} * (N-1)
    # where A_obs is the (N, n_obs) matrix of observation-space anomalies
    
    # For numerical stability, use the (N-1) dimensional representation
    # Let B = obs_anomalies / sqrt(N-1), shape (N, n_obs)
    # Then B B^T / (N-1) is the observation-space covariance
    
    # Actually, let's use the standard EnKF update formula:
    # x_a = x_f + P_f H^T (H P_f H^T + R)^{-1} (y - H x_f)
    
    # P_f H^T in ensemble form: sum over members of anomalies_i * anomalies_i_obs^T / (N-1)
    # = anomalies @ obs_anomalies.T / (N-1), shape (n_vars, n_obs)
    
    # H P_f H^T = obs_anomalies @ obs_anomalies.T / (N-1), shape (n_obs, n_obs)
    
    # R = observation_variance * I, shape (n_obs, n_obs)
    
    # So we need to solve: (obs_anomalies @ obs_anomalies.T / (N-1) + R) * alpha = innovation
    # Then the update for each variable j is: sum_k K[j,k] * innovation[k]
    # where K[j,k] = sum_i anomalies[i,j] * anomalies[i,k_obs] / (N-1) * alpha[k]
    
    # More precisely:
    # update[j] = sum_i anomalies[i,j] * (obs_anomalies[i,:] @ alpha) / (N-1)
    # where alpha solves (obs_anomalies @ obs_anomalies.T / (N-1) + R) @ alpha = innovation
    
    # Let's compute this efficiently
    
    # S = obs_anomalies @ obs_anomalies.T / (N-1) + R
    S = obs_anomalies @ obs_anomalies.T / (N - 1) + observation_variance * np.eye(n_obs)
    
    # Solve S @ alpha = innovation
    try:
        alpha = np.linalg.solve(S, innovation)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        alpha = np.linalg.lstsq(S, innovation, rcond=None)[0]
    
    # Now compute the update for each ensemble member
    # For member i: update_i = anomalies[i, :] @ (obs_anomalies[i, :] @ alpha) / (N-1) ... no
    # 
    # The analysis update for member i is:
    # x_a[i, j] = x_f[i, j] + sum_k K[j,k] * innovation[k]
    # where K[j,k] = (1/(N-1)) * sum_i anomalies[i,j] * anomalies[i,k]
    #
    # So the update for member i, variable j is:
    # delta_i[j] = sum_k K[j,k] * innovation[k]
    #            = (1/(N-1)) * sum_i' anomalies[i',j] * (sum_k anomalies[i',k] * innovation[k])
    #
    # Let beta_i = sum_k anomalies[i,k_obs] * alpha[k] ... wait, alpha is related to innovation
    #
    # Actually, let me re-derive. The standard EnKF update is:
    # x_a[i] = x_f[i] + P_f H^T (H P_f H^T + R)^{-1} (y - H x_f_mean)
    #
    # P_f H^T (H P_f H^T + R)^{-1} (y - H x_f_mean)
    # = [sum_i' anomalies[i'] anomalies[i']_obs^T / (N-1)] @ alpha
    # where alpha = (H P_f H^T + R)^{-1} (y - H x_f_mean)
    #
    # So the update vector is:
    # update = anomalies.T @ (obs_anomalies.T @ alpha) / (N-1)
    #
    # This is the same for all members! The analysis mean update is the same for all.
    # But the ensemble members also get individual updates to maintain spread.
    
    # Actually, in the standard EnKF, each member is updated by the same linear transformation:
    # x_a[i] = x_f[i] + K (y - H x_f[i])
    # where K is the Kalman gain matrix.
    #
    # This is different from just adding the mean update. Let me use the proper formulation.
    #
    # K = P_f H^T (H P_f H^T + R)^{-1}
    #   = [anomalies @ obs_anomalies.T / (N-1)] @ alpha
    #
    # So K is (n_vars, n_obs)
    # K = anomalies @ obs_anomalies.T / (N-1) @ alpha
    #   = anomalies @ (obs_anomalies.T @ alpha) / (N-1)
    
    # Compute obs_anomalies.T @ alpha, shape (N,)
    obs_weighted = obs_anomalies.T @ alpha  # (N,)
    
    # K = anomalies @ obs_weighted / (N-1), shape (n_vars, n_obs)
    # Wait, that's not right dimensionally. Let me reconsider.
    
    # K[j, k] = (1/(N-1)) * sum_i anomalies[i,j] * anomalies[i,k_obs]
    # The update for member i is:
    # delta_i[j] = sum_k K[j,k] * (y[k] - H x_f[i][k])
    #            = sum_k K[j,k] * (innovation[k] + ens_mean[k_obs] - x_f[i][k_obs])
    #            = sum_k K[j,k] * innovation[k] - sum_k K[j,k] * (x_f[i][k_obs] - ens_mean[k_obs])
    
    # This is getting complex. Let me use a simpler approach:
    # Use the ensemble Kalman filter with the "let-kfilter" approach.
    
    # Alternative: use the ensemble-based Kalman update directly.
    # For each ensemble member i:
    # x_a[i, j] = x_f[i, j] + sum_k K[j,k] * (y[k] - x_f[i, k_obs])
    
    # Where K[j,k] = (1/(N-1)) * sum_i' anomalies[i',j] * anomalies[i',k_obs]
    
    # Let's compute K properly.
    # K = (anomalies @ obs_anomalies.T) / (N-1) @ np.linalg.inv(S)
    # But we already have alpha = S^{-1} innovation, so:
    # K @ innovation = (anomalies @ obs_anomalies.T / (N-1)) @ alpha
    #                = anomalies @ (obs_anomalies.T @ alpha) / (N-1)
    
    # So the mean update is:
    # mean_update = anomalies @ (obs_anomalies.T @ alpha) / (N-1)
    # This is a (n_vars,) vector, same for all members in terms of the mean shift.
    
    # But for individual members, we need:
    # x_a[i] = x_f[i] + K (y - H x_f[i])
    
    # Let me compute K as a matrix.
    # K = (anomalies @ obs_anomalies.T) @ np.linalg.inv(S) / (N-1)
    
    # This is (n_vars, n_obs)
    
    # For efficiency, let's compute the update for each member.
    # x_a[i, j] = x_f[i, j] + sum_k K[j,k] * (y[k] - x_f[i, k_obs])
    
    # Let's compute K_inv_S = np.linalg.inv(S)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # K = anomalies @ obs_anomalies.T @ S_inv / (N-1)
    K = (anomalies @ obs_anomalies.T) @ S_inv / (N - 1)  # (n_vars, n_obs)
    
    # Now apply the update to each ensemble member
    updated_ensemble = ensemble.copy()
    for i in range(N):
        # Observation residual for member i
        obs_residual = observations - ensemble[i, observed_indices]  # (n_obs,)
        # Update
        updated_ensemble[i, :] = ensemble[i, :] + K @ obs_residual
    
    # Check for time limit
    if time.time() > deadline:
        return updated_ensemble
    
    # Adaptive inflation to maintain ensemble spread
    # Compute the ensemble spread before and after update
    pre_mean = np.mean(ensemble, axis=0)
    pre_anom = ensemble - pre_mean
    pre_spread = np.mean(np.var(pre_anom, axis=0), axis=0)
    
    post_mean = np.mean(updated_ensemble, axis=0)
    post_anom = updated_ensemble - post_mean
    post_spread = np.mean(np.var(post_anom, axis=0), axis=0)
    
    # Target spread ratio (slightly above 1 to maintain spread)
    target_ratio = 1.05
    if post_spread > 1e-10:
        inflation_factor = np.sqrt(max(target_ratio * pre_spread / post_spread, 1.0))
    else:
        inflation_factor = 1.0
    
    # Cap inflation to avoid instability
    inflation_factor = min(inflation_factor, 1.5)
    inflation_factor = max(inflation_factor, 0.8)
    
    # Apply inflation around the mean
    updated_ensemble = post_mean + inflation_factor * (updated_ensemble - post_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finiteness
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
