"""Rewritten program: Ensemble Kalman Filter (EnKF) with inflation and localization."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Kalman Filter (EnKF) analysis step.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed states
        observed_indices: indices of observed coordinates (even indices)
        observation_variance: scalar variance of observation noise
    
    Returns:
        updated ensemble with same (16, 20) shape
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observations)
    
    # Convert to float64 for numerical stability
    X = ensemble.astype(np.float64)
    y = observations.astype(np.float64)
    
    # Compute ensemble mean and covariance
    X_mean = np.mean(X, axis=0)
    
    # Anomaly matrix (centered ensemble)
    X_anom = X - X_mean
    
    # Observation operator H: maps state to observed coordinates
    # H is a selection matrix: H[i, j] = 1 if j == observed_indices[i]
    # We can work directly with observed indices
    
    # Compute observation-space covariance: S = H P H^T + R
    # P = (1/(n-1)) * X_anom @ X_anom.T
    # H P H^T = (1/(n-1)) * (X_anom[:, observed_indices]) @ (X_anom[:, observed_indices]).T
    
    X_obs_anom = X_anom[:, observed_indices]  # (n_members, n_obs)
    
    # Innovation covariance S = H P H^T + R
    # H P H^T = (1/(n-1)) * X_obs_anom @ X_obs_anom.T
    S = (X_obs_anom @ X_obs_anom.T) / (n_members - 1)
    S += observation_variance * np.eye(n_obs)
    
    # Innovation
    y_innov = y - X_mean[observed_indices]  # (n_obs,)
    
    # Kalman gain K = P H^T S^{-1}
    # P H^T: (n_vars, n_obs) = (1/(n-1)) * X_anom @ X_obs_anom.T
    PHt = (X_anom @ X_obs_anom.T) / (n_members - 1)
    
    # Solve S^T K^T = PHt^T  =>  K = S^{-1} PHt^T
    # More stable: solve S K^T = PHt^T using Cholesky
    try:
        L = np.linalg.cholesky(S)
        # Solve L K1^T = PHt^T => K1^T = L^{-1} PHt^T
        K1T = np.linalg.solve(L, PHt.T)
        # Solve L^T K^T = K1^T => K^T = (L^T)^{-1} K1^T
        KT = np.linalg.solve(L.T, K1T)
        K = KT.T  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        K = PHt @ np.linalg.pinv(S)
    
    # Apply localization: reduce influence of distant observations
    # For Lorenz-96 with periodic indices, use a Gaussian decay based on distance
    # Distance in the ring of 20 variables
    loc_width = 4.0  # localization width in grid points
    loc_matrix = np.ones((n_vars, n_obs))
    for i in range(n_vars):
        for j, obs_idx in enumerate(observed_indices):
            # Compute minimum distance on the ring
            dist = abs(i - obs_idx)
            dist = min(dist, n_vars - dist)
            loc_matrix[i, j] = np.exp(-0.5 * (dist / loc_width) ** 2)
    
    # Apply localization to Kalman gain
    K_loc = K * loc_matrix
    
    # Update ensemble
    # X_updated = X + K_loc @ (y_innov[None, :] - H(X - X_mean))
    # For each member m: X[m] += K_loc @ (y - X_mean[obs]) - K_loc @ (X[m, obs] - X_mean[obs])
    #                   = X[m] + K_loc @ (y - X[m, obs])
    
    # Compute innovation for each member
    # y - X[m, obs] for all members: (n_members, n_obs)
    member_innov = y[None, :] - X[:, observed_indices]  # (n_members, n_obs)
    
    # Update
    X_updated = X + member_innov @ K_loc.T
    
    # Adaptive inflation: detect if ensemble spread is too small
    # Compute pre-update and post-update spread at observed points
    pre_spread = np.std(X[:, observed_indices], axis=0)
    post_spread = np.std(X_updated[:, observed_indices], axis=0)
    
    # Inflate if post-update spread is significantly smaller than pre-update
    # Target: maintain spread at least 0.5x of pre-update (or some threshold)
    min_spread_ratio = 0.5
    inflation_factor = 1.0
    
    for i in range(n_obs):
        if pre_spread[i] > 1e-6:
            ratio = post_spread[i] / pre_spread[i]
            if ratio < min_spread_ratio:
                # Need to inflate to restore spread
                required_ratio = min_spread_ratio
                # After inflation by factor alpha around mean:
                # new_spread = alpha * post_spread
                # We want alpha * post_spread >= required_ratio * pre_spread
                alpha_i = required_ratio * pre_spread[i] / post_spread[i]
                inflation_factor = max(inflation_factor, alpha_i)
    
    # Apply inflation around ensemble mean
    if inflation_factor > 1.0 + 1e-6:
        X_updated_mean = np.mean(X_updated, axis=0)
        X_updated = X_updated_mean + inflation_factor * (X_updated - X_updated_mean)
    
    # Clip to ensure finite values within reasonable bounds
    X_updated = np.clip(X_updated, -100.0, 100.0)
    
    # Ensure no NaN or Inf
    if not np.all(np.isfinite(X_updated)):
        X_updated = np.nan_to_num(X_updated, nan=0.0, posinf=100.0, neginf=-100.0)
        X_updated = np.clip(X_updated, -100.0, 100.0)
    
    return X_updated
# EVOLVE-BLOCK-END
