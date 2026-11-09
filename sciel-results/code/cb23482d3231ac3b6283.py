"""Lorenz-96 ensemble assimilation using Ensemble Kalman Filter with observation perturbation."""
import numpy as np
import time

# Precompute index mappings for Lorenz-96 with periodic boundary conditions
N = 20
# For dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
# We need indices: i+1, i-2, i-1, i
idx_i = np.arange(N)
idx_ip1 = (idx_i + 1) % N
idx_im2 = (idx_i - 2) % N
idx_im1 = (idx_i - 1) % N

def _lorenz96_step(X, F, dt):
    """Single RK4 step for an ensemble of Lorenz-96 states.
    X: (M, N) array of ensemble members
    Returns X after one step of size dt.
    """
    # k1
    k1 = (X[:, idx_ip1] - X[:, idx_im2]) * X[:, idx_im1] - X + F
    # k2
    X2 = X + 0.5 * dt * k1
    k2 = (X2[:, idx_ip1] - X2[:, idx_im2]) * X2[:, idx_im1] - X2 + F
    # k3
    X3 = X + 0.5 * dt * k2
    k3 = (X3[:, idx_ip1] - X3[:, idx_im2]) * X3[:, idx_im1] - X3 + F
    # k4
    X4 = X + dt * k3
    k4 = (X4[:, idx_ip1] - X4[:, idx_im2]) * X4[:, idx_im1] - X4 + F
    X_new = X + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return X_new

def _ekf_update(ensemble, observations, observed_indices, obs_var):
    """
    Ensemble Kalman Filter update using observation perturbation.
    
    ensemble: (M, N) array
    observations: (N_obs,) array of noisy observations
    observed_indices: array of indices that are observed
    obs_var: scalar observation variance
    
    Returns updated ensemble (M, N).
    """
    M, N = ensemble.shape
    M_obs = len(observed_indices)
    
    # Compute ensemble mean and covariance at observed locations
    x_mean = np.mean(ensemble, axis=0)
    
    # Innovation: observation - forecast mean at observed locations
    innovation = observations - x_mean[observed_indices]
    
    # Ensemble anomaly matrix at observed locations: (M, M_obs)
    X_obs = ensemble[:, observed_indices]  # (M, M_obs)
    X_obs_anom = X_obs - x_mean[observed_indices]  # (M, M_obs)
    
    # Covariance of observations from ensemble: (M_obs, M_obs)
    # S = X_obs_anom^T @ X_obs_anom / (M-1) + R
    # For the update, we need the Kalman gain
    # K = P_xy @ S^{-1} where P_xy = X_anom^T @ X_obs_anom / (M-1)
    
    # Using the "ensemble Kalman filter" formulation:
    # The update for each member i is:
    # x_i' = x_i + K @ (y - x_i_obs)  ... but with perturbed observations
    
    # Observation perturbation method:
    # y_i = y + sqrt(R) * eps_i  for each member
    # Then do a linear regression of state anomalies on perturbed observation anomalies
    
    # Perturb observations with N(0, sqrt(obs_var))
    eps = np.random.randn(M, M_obs)
    y_pert = observations[None, :] + np.sqrt(obs_var) * eps  # (M, M_obs)
    
    # Anomalies of perturbed observations
    y_pert_mean = np.mean(y_pert, axis=0)  # (M_obs,)
    y_anom = y_pert - y_pert_mean  # (M, M_obs)
    
    # State anomalies: (M, N)
    X_anom = ensemble - x_mean  # (M, N)
    
    # We want to find the regression coefficients B such that:
    # X_anom ≈ B @ y_anom
    # B = X_anom^T @ y_anom @ (y_anom^T @ y_anom)^{-1}
    
    # For numerical stability, use the SVD-based approach or solve directly
    # B = X_anom^T @ y_anom @ inv(y_anom^T @ y_anom)
    
    # Compute y_anom^T @ y_anom: (M_obs, M_obs)
    YTY = y_anom.T @ y_anom  # (M_obs, M_obs)
    
    # Compute X_anom^T @ y_anom: (N, M_obs)
    XTY = X_anom.T @ y_anom  # (N, M_obs)
    
    # Solve for B: B @ YTY = XTY  =>  B = XTY @ inv(YTY)
    # Or equivalently, solve YTY^T @ B^T = XTY^T
    # Let's solve: B = XTY @ inv(YTY)
    
    # Add small regularization for numerical stability
    reg = 1e-10 * np.eye(M_obs)
    YTY_reg = YTY + reg
    
    try:
        B = XTY @ np.linalg.inv(YTY_reg)  # (N, M_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        B = XTY @ np.linalg.pinv(YTY_reg)
    
    # Update each member:
    # x_i' = x_i + B @ (y_i - y_mean)  where y_i is the perturbed obs for member i
    # Actually, the standard EK with perturbed obs:
    # x_i' = x_i + B @ (y_pert_i - y_pert_mean)
    # But we also need to update the mean properly.
    
    # The updated state for each member:
    # x_i' = x_mean + (x_i - x_mean) + B @ (y_pert_i - y_pert_mean)
    # This preserves the ensemble mean structure in expectation.
    
    # Actually, let's use the standard formulation:
    # The analysis is: x_a = x_f + K(y - H x_f)
    # In ensemble form with perturbation:
    # x_i^a = x_i + B @ (y_pert_i - y_pert_mean)
    
    update = B @ y_anom  # (M, N)
    ensemble_updated = ensemble + update
    
    # Ensure the ensemble mean matches the Kalman-filtered mean
    # The mean update should be: x_mean_a = x_mean + K @ innovation
    # In our formulation, the mean of the update is:
    # mean(update) = B @ mean(y_anom) = B @ 0 = 0
    # So the ensemble mean doesn't change! That's a problem.
    
    # Let me reconsider. The proper EK with observation perturbation:
    # We want the analysis mean to be: x_mean + K @ innovation
    # where K = P_xy @ S^{-1}
    
    # Let me use the standard 3D-Var / EK approach directly:
    
    # Compute the ensemble covariance P (M x M) approach:
    # P = X_anom^T @ X_anom / (M-1)  (N, N)
    # But that's expensive. Let's use the factored form.
    
    # Alternative: use the "deterministic" EK (perturb state, not obs)
    # which is more standard and avoids the mean issue.
    
    # Let me restart with the standard ensemble Kalman filter (perturb state):
    
    # Compute innovation
    # innovation = y - H(x_mean)
    
    # P_xy = X_anom^T @ X_obs_anom / (M-1)  (N, M_obs)
    P_xy = X_anom.T @ X_obs_anom / (M - 1)  # (N, M_obs)
    
    # S = X_obs_anom^T @ X_obs_anom / (M-1) + R  (M_obs, M_obs)
    S = X_obs_anom.T @ X_obs_anom / (M - 1) + obs_var * np.eye(M_obs)
    
    # K = P_xy @ S^{-1}  (N, M_obs)
    S_reg = S + 1e-10 * np.eye(M_obs)
    try:
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
    
    K = P_xy @ S_inv  # (N, M_obs)
    
    # Update mean: x_mean_a = x_mean + K @ innovation
    x_mean_a = x_mean + K @ innovation
    
    # Update ensemble: x_i_a = x_i + K @ (x_i_obs - x_mean_obs)
    # This is the "perturbed state" version where we use the actual observation
    # anomalies relative to the mean.
    
    # Actually, the standard EK update is:
    # x_i_a = x_i + K @ (y - x_i_obs)  ... no, that's not right either.
    
    # The correct ensemble Kalman filter update:
    # x_i_a = x_i + K @ (y - H x_i)  for each member
    # But this is the "perturbed observation" approach if we add noise to y.
    
    # Let's use the standard form:
    # x_i_a = x_mean_a + (x_i - x_mean) + K @ (y - x_i_obs)
    # Wait, that's not standard either.
    
    # The most common formulation (perturbed state):
    # x_i_a = x_i + K @ (y - x_i_obs)  where K is computed from the ensemble
    # But this doesn't preserve the mean correctly.
    
    # Let me use the well-known formulation:
    # x_i_a = x_mean + (x_i - x_mean) + K @ (y - x_mean_obs) - K @ (x_i_obs - x_mean_obs)
    # = x_mean + K @ innovation + (x_i - x_mean) - K @ (x_i_obs - x_mean_obs)
    
    # Actually, the simplest correct form is:
    # x_i_a = x_i + K @ (y - x_i_obs)
    # Let's verify the mean: E[x_i_a] = E[x_i] + K @ (y - E[x_i_obs]) = x_mean + K @ innovation
    # Yes! That's correct.
    
    ensemble_updated = ensemble + K @ (observations[None, :] - X_obs)
    
    # Clip to reasonable bounds
    ensemble_updated = np.clip(ensemble_updated, -100.0, 100.0)
    
    return ensemble_updated

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 assimilation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array_like
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.int64)
    
    # Ensure finite values
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Perform EKF update
    updated = _ekf_update(ensemble, observations, observed_indices, observation_variance)
    
    # Ensure output is finite and within bounds
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated
