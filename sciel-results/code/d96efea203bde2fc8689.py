"""Ensemble Kalman Filter with localization and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with:
    - Ensemble mean-based covariance
    - Gaspari-Cohn localization (correlation distance ~2.5)
    - Adaptive multiplicative inflation with bounds
    - Perturbed observations (EnSRF)
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_members = ensemble.shape[0]  # 16
    n_state = ensemble.shape[1]    # 20
    n_obs = len(observed_indices)  # 10
    
    # --- Localization: Gaspari-Cohn correlation function ---
    def gaspari_cohn(d, L=2.5):
        """Gaspari-Cohn correlation function with correlation length L."""
        d = np.abs(d)
        result = np.zeros_like(d, dtype=np.float64)
        mask1 = d < L / 2.0
        mask2 = (d >= L / 2.0) & (d <= 1.5 * L)
        # For d < L/2
        x1 = d[mask1] / L
        result[mask1] = (-25.0 / 2.0 * x1**5 + 50.0 * x1**4 - 30.0 * x1**3 + 1.0)
        # For L/2 <= d <= 1.5L
        x2 = (d[mask2] - 1.5 * L) / L
        result[mask2] = (25.0 / 2.0 * x2**5 - 25.0 * x2**4 + 75.0 / 2.0 * x2**3 + 4.0 * x2 - 5.0 / 12.0)
        # For d > 1.5L, result stays 0
        return result
    
    # Build localization matrix for observed variables
    # obs_indices are the observed coordinates
    loc_matrix = np.zeros((n_obs, n_obs), dtype=np.float64)
    for i in range(n_obs):
        for j in range(n_obs):
            loc_matrix[i, j] = gaspari_cohn(observed_indices[i] - observed_indices[j], L=2.5)
    
    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (n_members, n_state)
    
    # --- Observation innovation ---
    # Ensemble mean at observed indices
    obs_ens_mean = ens_mean[observed_indices]  # (n_obs,)
    innovation = observations - obs_ens_mean   # (n_obs,)
    
    # --- Compute ensemble covariance at observed indices ---
    # C_oo = (1/(n-1)) * A_obs @ A_obs.T
    A_obs = anomalies[:, observed_indices]  # (n_members, n_obs)
    C_oo = (A_obs @ A_obs.T) / (n_members - 1)  # (n_obs, n_obs)
    
    # Localize the covariance
    C_oo_loc = C_oo * loc_matrix  # element-wise localization
    
    # --- Adaptive inflation ---
    # Estimate inflation factor from innovation statistics
    # Target: innovation variance should match C_oo + R
    R = observation_variance * np.eye(n_obs)
    expected_innov_var = np.diag(C_oo_loc) + observation_variance  # (n_obs,)
    actual_innov_var = innovation ** 2  # (n_obs,)
    
    # Average ratio
    ratio = np.mean(actual_innov_var / np.maximum(expected_innov_var, 1e-10))
    # Inflation factor: sqrt of ratio, bounded
    inflation = np.sqrt(np.clip(ratio, 0.95, 1.5))
    # Smooth inflation: blend with 1.0 to avoid over-correction
    inflation = 0.7 * inflation + 0.3 * 1.0
    
    # Apply inflation to anomalies
    inflated_anomalies = anomalies * inflation
    inflated_ens_mean = ens_mean.copy()
    
    # Recompute C_oo with inflated anomalies
    A_obs_inf = inflated_anomalies[:, observed_indices]
    C_oo_inf = (A_obs_inf @ A_obs_inf.T) / (n_members - 1)
    C_oo_inf_loc = C_oo_inf * loc_matrix
    
    # --- Kalman gain computation ---
    # S = C_oo + R
    S = C_oo_inf_loc + R  # (n_obs, n_obs)
    
    # Invert S (should be positive definite)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # K = C_xo @ S_inv
    # C_xo = (1/(n-1)) * A @ A_obs.T  (n_state, n_obs)
    A_all = inflated_anomalies  # (n_members, n_state)
    C_xo = (A_all @ A_obs_inf.T) / (n_members - 1)  # (n_state, n_obs)
    
    K = C_xo @ S_inv  # (n_state, n_obs)
    
    # --- Update ensemble ---
    # For each member, update using the same K but with perturbed observations (EnSRF)
    # Alternatively, use the deterministic update with mean shift
    # Here we use the standard EnKF update:
    # x_a = x_b + K @ (y - H(x_b))
    # But for ensemble, we apply the same gain to each member's innovation
    
    # Perturb observations for each member (EnSRF approach)
    rng = np.random.default_rng(seed=42)
    updated_ensemble = np.zeros_like(ensemble)
    
    for m in range(n_members):
        # Perturbed observation for this member
        obs_pert = observations + rng.standard_normal(n_obs) * np.sqrt(observation_variance)
        innov_m = obs_pert - ensemble[m, observed_indices]
        updated_ensemble[m, :] = ensemble[m, :] + K @ innov_m
    
    # Re-center the ensemble to maintain the correct mean
    # The EnSRF update should preserve the mean if done correctly, but let's enforce it
    updated_mean = np.mean(updated_ensemble, axis=0)
    target_mean = ens_mean + K @ innovation
    updated_ensemble += (target_mean - updated_mean)
    
    # Clip to reasonable bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finiteness
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
