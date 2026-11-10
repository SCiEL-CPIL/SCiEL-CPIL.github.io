"""Ensemble Kalman Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# Gaspari-Cohn correlation function for localization
def _gaspari_cohn(d, L):
    """Gaspari-Cohn quintic correlation function. d = distance, L = correlation length."""
    d = np.abs(d)
    L = max(L, 1e-10)
    result = np.zeros_like(d, dtype=float)
    # Region 1: |d| <= L/2
    mask1 = d <= L / 2.0
    d1 = d[mask1]
    d1_L = d1 / L
    result[mask1] = (-4.0 * d1_L**5 + 6.0 * d1_L**4 + 5.0 * d1_L**3 - 2.0 * d1_L**2) / 12.0
    # Region 2: L/2 < |d| <= 3L/2
    mask2 = (d > L / 2.0) & (d <= 3.0 * L / 2.0)
    d2 = d[mask2]
    d2_L = d2 / L
    result[mask2] = (2.0 * d2_L**5 - 5.0 * d2_L**4 + -4.0 * d2_L**3 + 10.0 * d2_L**2 + 5.0 * d2_L - 5.0 / 2.0) / 12.0
    # Region 3: 3L/2 < |d| <= 5L/2
    mask3 = (d > 3.0 * L / 2.0) & (d <= 5.0 * L / 2.0)
    d3 = d[mask3]
    d3_L = d3 / L
    result[mask3] = (-2.0 * d3_L**5 + 25.0 * d3_L**4 - 120.0 * d3_L**3 + 220.0 * d3_L**2 - 150.0 * d3_L + 25.0 / 2.0) / 12.0
    # Beyond 5L/2: zero
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each a 20-state vector.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array_like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    obs_idx = np.array(observed_indices)
    
    # Convert to float64 for numerical stability
    X = ensemble.astype(np.float64).copy()
    y = observations.astype(np.float64).copy()
    R = observation_variance  # scalar variance
    
    # Step 1: Compute ensemble mean and anomalies
    X_mean = np.mean(X, axis=0)
    X_anom = X - X_mean  # (n_members, n_state)
    
    # Step 2: Compute forecast observation anomalies
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    Y_anom = X_anom[:, obs_idx]  # (n_members, n_obs)
    
    # Step 3: Compute Kalman gain
    # S = H P H^T + R  where P is ensemble covariance
    # Using sample covariance: P = (1/(N-1)) * X_anom^T X_anom
    # S = (1/(N-1)) * Y_anom^T Y_anom + R*I
    
    # Compute Y_anom^T Y_anom (n_obs x n_obs)
    YYT = Y_anom.T @ Y_anom  # (n_obs, n_obs)
    S = YYT / (n_members - 1) + R * np.eye(n_obs)
    
    # Cholesky decomposition of S for numerical stability
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition with regularization
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Innovation: y - H X_mean
    innovation = y - X_mean[obs_idx]
    
    # Kalman gain: K = P H^T S^-1 = (1/(N-1)) X_anom^T Y_anom S^-1
    # Compute X_anom^T Y_anom (n_state x n_obs)
    XYT = X_anom.T @ Y_anom  # (n_state, n_obs)
    
    # K = XYT / (N-1) * S^-1
    # Instead of explicit inverse, solve: K = XYT / (N-1) * inv(S)
    # Using Cholesky: inv(S) = L^-T L^-1
    # K = XYT / (N-1) * L^-T L^-1
    # Solve L^T Z = XYT/(N-1) then L K = Z
    
    XYT_scaled = XYT / (n_members - 1)  # (n_state, n_obs)
    
    # Solve L^T Z = XYT_scaled  => Z = L^-T XYT_scaled
    Z = np.linalg.solve(L.T, XYT_scaled.T).T  # (n_state, n_obs)
    # Solve L K = Z  => K = L^-1 Z
    K = np.linalg.solve(L, Z.T).T  # (n_state, n_obs)
    
    # Step 4: Apply localization to cross-covariance
    # Localize the Kalman gain: for each state i and obs j,
    # multiply K[i,j] by rho(|i - obs_idx[j]| / L)
    L_corr = 3.0  # Correlation length in state space
    obs_positions = obs_idx.astype(float)
    state_positions = np.arange(n_state, dtype=float)
    
    # Build localization matrix (n_state x n_obs)
    loc_matrix = np.zeros((n_state, n_obs))
    for j in range(n_obs):
        distances = np.abs(state_positions - obs_positions[j])
        loc_matrix[:, j] = _gaspari_cohn(distances, L_corr)
    
    K_localized = K * loc_matrix
    
    # Step 5: Compute analysis update
    # X_a = X + K_localized @ (y - H X)  for each member
    # Innovation is the same for all members (deterministic update)
    update = K_localized @ innovation  # (n_state,)
    
    X_analysis = X + update[np.newaxis, :]  # (n_members, n_state)
    
    # Step 6: Adaptive inflation to maintain ensemble spread
    # Compute pre-analysis and post-analysis spread
    X_mean_pre = np.mean(X, axis=0)
    X_mean_post = np.mean(X_analysis, axis=0)
    
    # Pre-analysis ensemble variance (trace of covariance)
    X_anom_pre = X - X_mean_pre
    var_pre = np.mean(np.sum(X_anom_pre**2, axis=1))
    
    # Post-analysis ensemble variance
    X_anom_post = X_analysis - X_mean_post
    var_post = np.mean(np.sum(X_anom_post**2, axis=1))
    
    # Target: maintain at least some fraction of pre-analysis spread
    # Inflation factor alpha such that var_post * alpha^2 >= target_var
    target_var = 0.85 * var_pre  # Keep at least 85% of pre-analysis spread
    if var_post > 1e-12:
        alpha = np.sqrt(max(target_var / var_post, 1.0))
    else:
        alpha = 1.0
    
    # Cap inflation to avoid runaway
    alpha = min(alpha, 1.5)
    
    # Apply inflation around the ensemble mean
    if alpha > 1.0 + 1e-6:
        X_analysis = X_mean_post + alpha * (X_analysis - X_mean_post)
    
    # Step 7: Stochastic perturbation to maintain ensemble diversity
    # Add small random perturbations to prevent ensemble collapse
    # The perturbation should be consistent with the observation noise scale
    perturbation_scale = np.sqrt(R) * 0.1  # Small perturbation
    rng = np.random.default_rng()
    perturbations = rng.standard_normal((n_members, n_state)) * perturbation_scale
    
    # Apply perturbations only to unobserved components to not violate observations
    # Actually, apply to all but scale down for observed
    X_analysis += perturbations
    
    # Ensure observed components stay close to observations
    # Re-nudge observed components slightly
    for i, idx in enumerate(obs_idx):
        residual = y[i] - X_analysis[:, idx]
        # Gentle nudge to keep observed components near observations
        X_analysis[:, idx] += 0.3 * residual
    
    # Step 8: Clip to valid range
    X_analysis = np.clip(X_analysis, -100.0, 100.0)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(X_analysis)):
        # Fallback to simple nudging if something went wrong
        X_analysis = X.copy()
        X_analysis[:, obs_idx] += 0.6 * (y - X[:, obs_idx])
        X_analysis = np.clip(X_analysis, -100.0, 100.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Return what we have
        return X_analysis.astype(np.float64)
    
    return X_analysis.astype(np.float64)
