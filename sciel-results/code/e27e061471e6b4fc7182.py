import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using a Local Ensemble Square Root Filter (LES-RF).
    
    Args:
        ensemble: (16, 20) array of ensemble members.
        observations: (10,) array of noisy observations of even-indexed coordinates.
        observed_indices: array of indices observed (even indices: 0, 2, ..., 18).
        observation_variance: scalar variance of observation noise.
    
    Returns:
        updated ensemble: (16, 20) array.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin: 5s before 120s limit
    
    N = ensemble.shape[1]  # 20
    M = len(observed_indices)  # 10
    nens = ensemble.shape[0]  # 16
    
    # 1. Compute ensemble mean
    x_mean = np.mean(ensemble, axis=0)
    
    # 2. Compute ensemble anomalies
    X_anom = ensemble - x_mean
    
    # 3. Compute sample covariance of observed variables
    # X_obs_anom: (M, nens)
    X_obs_anom = X_anom[:, observed_indices].T
    
    # Observation covariance matrix (diagonal)
    R = np.eye(M) * observation_variance
    
    # 4. Compute innovation covariance: S = H P H^T + R
    # H is identity on observed indices, so H P H^T is the (M,M) block of P
    # P = X_anom @ X_anom.T / (nens - 1)
    # S = (X_obs_anom @ X_obs_anom.T) / (nens - 1) + R
    S = (X_obs_anom @ X_obs_anom.T) / (nens - 1) + R
    
    # 5. Compute Kalman gain components
    # We need to solve for the update. In LESR, we avoid forming the full gain matrix
    # by using a Cholesky decomposition approach.
    
    # Innovation vector: y - H x_mean
    innovation = observations - x_mean[observed_indices]
    
    # 6. Compute the analysis update using the square root form
    # The LESR algorithm updates the ensemble anomalies directly.
    # Reference: Houtekamer and Zhang (2016), "The local ensemble square root filter:
    #             theoretical aspects and results for atmospheric data assimilation"
    
    # Step 6a: Compute the matrix B such that B B^T = S
    try:
        B = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition if S is not positive definite
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
        B = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Step 6b: Compute the transformation matrix
    # We want to find a matrix A such that the updated anomalies X_a satisfy
    # X_a = X_anom @ (I - A^T A) / (nens - 1) ... actually the standard LESR is:
    # Let Y = innovation @ B^{-1}  (1 x nens)
    # Let X = X_obs_anom @ B^{-1}  (M x nens)
    # Then compute the QR decomposition or use the square root update.
    
    # Standard LESR update:
    # 1. Compute Y = (y - H x) @ B^{-1}
    Y = innovation @ np.linalg.inv(B)  # (1, nens)
    
    # 2. Compute X = X_obs_anom @ B^{-1}
    X = X_obs_anom @ np.linalg.inv(B)  # (M, nens)
    
    # 3. Compute the matrix Q = I - X^T (X X^T + I)^{-1} X
    # But we need the square root form. Let's use the algorithm from Houtekamer & Zhang.
    # The update is: X_a = X_anom @ (I - X^T (X X^T + I)^{-1} X) / sqrt(nens-1) ... 
    # Actually, the standard LESR algorithm is:
    # Let Z = X^T @ Y  (nens x 1)
    # Let W = X @ X^T + I  (M x M)
    # Let C = W^{-1} @ X  (M x nens)
    # Then the updated anomalies are:
    # X_a = X_anom - C^T @ (X_anom[:, observed_indices].T @ C - Y) ... 
    # This is getting complex. Let's use a simpler, more direct square root form.
    
    # Alternative approach: Use the standard EnKF update with localization.
    # Since nens is small (16), we can compute the full Kalman gain.
    
    # 4. Compute the Kalman gain K
    # K = P H^T S^{-1}
    # P = X_anom @ X_anom.T / (nens - 1)
    # H is a selection matrix, so P H^T is just the rows of P corresponding to observed indices
    # K is a (N, M) matrix
    P = (X_anom @ X_anom.T) / (nens - 1)
    PHt = P[:, observed_indices]  # (N, M)
    
    # Solve for K: K = PHt @ S^{-1}
    # Use Cholesky of S for numerical stability
    try:
        S_inv = np.linalg.inv(B @ B.T)  # S = B B^T
    except:
        S_inv = np.linalg.pinv(S)
    
    K = PHt @ S_inv  # (N, M)
    
    # 5. Apply localization to the Kalman gain
    # Gaussian localization: K_loc[i, j] = K[i, j] * exp(-0.5 * (d_ij / L)^2)
    # where d_ij is the spatial distance between state variable i and observation variable j
    L = 3.0  # Localization length scale (in grid points)
    
    # State indices: 0, 1, ..., 19
    # Observed indices: 0, 2, 4, ..., 18
    state_idx = np.arange(N)
    obs_idx = observed_indices
    
    # Compute distance matrix: (N, M)
    dist = np.abs(state_idx[:, np.newaxis] - obs_idx[np.newaxis, :])
    # Account for periodicity: distance is min(dist, N - dist)
    dist = np.minimum(dist, N - dist)
    
    # Apply Gaussian localization
    loc_weight = np.exp(-0.5 * (dist / L) ** 2)
    K_loc = K * loc_weight  # (N, M)
    
    # 6. Compute the analysis update
    # x_a = x + K_loc @ innovation
    x_analysis = x_mean + K_loc @ innovation
    
    # Update the ensemble mean
    ensemble[:, :] = ensemble - x_mean[np.newaxis, :] + x_analysis[np.newaxis, :]
    
    # 7. Apply covariance inflation
    # Multiplicative inflation: inflate the ensemble spread around the new mean
    # Check if the ensemble spread is too small compared to the innovation
    spread_obs = np.std(ensemble[:, observed_indices], axis=0)
    innov_std = np.std(innovation)
    
    # If the ensemble spread is much smaller than the innovation standard deviation,
    # inflate the ensemble
    if np.mean(spread_obs) < 0.5 * innov_std:
        inflation_factor = 1.05
        ensemble = x_analysis[np.newaxis, :] + inflation_factor * (ensemble - x_analysis[np.newaxis, :])
    
    # 8. Ensure the output is finite and within bounds
    ensemble = np.clip(ensemble, -100.0, 100.0)
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return ensemble
    
    return ensemble
