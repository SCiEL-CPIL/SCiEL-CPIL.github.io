import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Performs a localized Ensemble Kalman Filter (EnKF) update.
    
    Parameters:
    ensemble: np.ndarray of shape (16, 20). Current ensemble states.
    observations: np.ndarray of shape (10,). Noisy observations of even indices.
    observed_indices: np.ndarray of shape (10,). Indices of observed variables.
    observation_variance: float. Variance of observation noise.
    
    Returns:
    np.ndarray of shape (16, 20). Updated ensemble states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Compute localized background covariance
    # Use Gaussian localization with sigma=3.0
    # R is the localization matrix, R[i,j] = exp(-d^2 / (2*sigma^2))
    sigma = 3.0
    # Compute pairwise distances on a ring of size D
    # Distance d(i,j) = min(|i-j|, D-|i-j|)
    i_idx = np.arange(D)
    j_idx = np.arange(D)
    dist = np.abs(i_idx[:, None] - j_idx[None, :])
    dist = np.minimum(dist, D - dist)
    R = np.exp(-dist**2 / (2 * sigma**2))
    
    # Localize the covariance: C = A^T A (N-1)
    # To apply localization efficiently without forming the full DxD covariance 
    # explicitly in a way that breaks the square-root structure, we can 
    # approximate by localizing the anomalies directly or using a diagonal 
    # approximation. However, for D=20, full covariance is cheap.
    
    # C = (1/(N-1)) * anomalies.T @ anomalies
    C = (anomalies.T @ anomalies) / (N - 1)
    
    # Apply localization to covariance
    C_loc = C * R
    
    # Ensure symmetry and positive semi-definiteness (add small jitter if needed)
    C_loc = (C_loc + C_loc.T) / 2.0
    
    # 3. Observation operator H (selection of indices)
    # H is effectively a selection matrix. 
    # H C H^T is the covariance of the observations.
    # H C H^T = C_loc[np.ix_(obs_idx, obs_idx)]
    HCH = C_loc[np.ix_(obs_idx, obs_idx)]
    
    # Observation covariance R_obs = var * I
    R_obs = observation_variance * np.eye(n_obs)
    
    # Innovation covariance S = H C H^T + R_obs
    S = HCH + R_obs
    
    # 4. Compute Kalman Gain K
    # K = C H^T S^-1
    # C H^T is a D x n_obs matrix: C_loc[:, obs_idx]
    CHt = C_loc[:, obs_idx]
    
    # Solve S^-1 CHt^T? No, K = CHt @ inv(S)
    # For efficiency and stability, solve S X = CHt^T then K = X^T
    # Or use Cholesky of S
    try:
        L_s = np.linalg.cholesky(S)
        # Solve L_s Y = CHt^T
        Y = np.linalg.solve(L_s, CHt.T)
        # Solve L_s^T X = Y
        X = np.linalg.solve(L_s.T, Y)
        K = X.T  # K is D x n_obs
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        K = CHt @ np.linalg.pinv(S)
        
    # 5. Compute Innovation
    # y = observations - H x_mean
    y = observations - mean[obs_idx]
    
    # 6. Update Mean
    mean_new = mean + K @ y
    
    # 7. Update Anomalies (Square Root Form)
    # a_new = a (I - K H)
    # H is selection, so K H is D x D matrix where columns obs_idx are K, others 0.
    # Let's construct the gain matrix G = K @ H
    # H is n_obs x D. K is D x n_obs.
    # G = K @ H is D x D.
    # G[:, obs_idx] = K
    # G[:, ~obs_idx] = 0
    G = np.zeros((D, D))
    G[:, obs_idx] = K
    
    # Update anomalies
    # a_new = a @ (I - G^T) ? 
    # Let's check dimensions.
    # a is N x D.
    # We want a_new = a - a @ G^T
    # Because x = x_mean + a.
    # x_new = x_mean_new + a_new
    # x_new = (x_mean + K y) + (a - a G^T)  -- wait, standard EnKF is:
    # x_new = x + K (y - H x)
    # x_new = x + K y - K H x
    # x_new = x + K y - K H (x_mean + a)
    # x_new = (x_mean + K y) + (a - K H a)  -- this is not quite right for anomalies.
    
    # Standard square root update:
    # Let a be the matrix of anomalies (N x D).
    # a_new = a (I - K H)
    # Let's verify:
    # x_new = x_mean_new + a_new
    # x_mean_new = x_mean + K y
    # a_new = a - a K H  (if we define K H acting on the right)
    # Actually, the update for each member i is:
    # x_i_new = x_i + K (y - H x_i)
    # x_i_new = x_i + K y - K H x_i
    # x_i_new = (x_i - K H x_i) + K y
    # Let's express in terms of mean and anomaly.
    # x_i = x_mean + a_i
    # x_i_new = x_mean + a_i - K H (x_mean + a_i) + K y
    # x_i_new = (x_mean - K H x_mean + K y) + (a_i - K H a_i)
    # x_i_new = (x_mean + K (y - H x_mean)) + a_i (I - K H)^T ?
    # Note: K H is D x D. a_i is 1 x D.
    # a_i (I - K H) is 1 x D.
    # So a_new_i = a_i (I - K H)
    # And mean_new = mean + K y
    
    # So we need to compute (I - G) where G = K H.
    # G is D x D.
    # a_new = a @ (I - G)
    
    I_G = np.eye(D) - G
    a_new = anomalies @ I_G
    
    # 8. Adaptive Inflation
    # Check spread ratio
    # Target spread ratio ~ 1.0 (or slightly higher to avoid collapse)
    # Current spread: std of a_new
    # Background spread: std of anomalies
    # Observation spread: std of y (not directly comparable)
    
    # Simple inflation: multiply anomalies by a factor if spread is too small.
    # Calculate spread ratio: sqrt(trace(C_new)/trace(C_old))
    # C_new = a_new.T @ a_new / (N-1)
    # C_old = C (or C_loc? usually background)
    
    # To keep it fast and robust, use a fixed small inflation if we detect collapse, 
    # or a dynamic one based on the ratio of background spread to analysis spread.
    
    # Let's use a simple adaptive inflation factor.
    # If the analysis spread is less than the background spread by a certain margin, inflate.
    # spread_bg = sqrt(np.mean(np.var(anomalies, axis=0)))
    # spread_an = sqrt(np.mean(np.var(a_new, axis=0)))
    # ratio = spread_an / spread_bg
    # if ratio < 0.9: inflate by 1.02
    
    # For simplicity and stability in this chaotic system, a constant small inflation 
    # is often better than aggressive adaptive schemes that can oscillate.
    # However, the prompt suggests "learned" or adaptive.
    # Let's try a gentle adaptive inflation.
    
    spread_bg = np.sqrt(np.mean(np.var(anomalies, axis=0)))
    spread_an = np.sqrt(np.mean(np.var(a_new, axis=0)))
    
    if spread_bg > 1e-6:
        ratio = spread_an / spread_bg
        if ratio < 0.95:
            # Inflate to bring spread back to ~0.95 of background
            factor = 0.95 / ratio
            # Cap inflation to prevent explosion
            factor = min(factor, 1.1)
            a_new *= factor
        elif ratio > 1.05:
            # Deflate if spread is too large (rare in EnKF without prior inflation)
            factor = 1.05 / ratio
            factor = max(factor, 0.9)
            a_new *= factor
    
    # 9. Reconstruct Ensemble
    ensemble_new = mean_new[None, :] + a_new
    
    # 10. Clip to valid range
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    # Ensure finite
    if not np.all(np.isfinite(ensemble_new)):
        # Fallback to original ensemble if something went wrong
        return ensemble.copy()
        
    return ensemble_new
# EVOLVE-BLOCK-END
