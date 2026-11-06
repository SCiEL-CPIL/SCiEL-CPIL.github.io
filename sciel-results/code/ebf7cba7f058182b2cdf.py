import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with Localization.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    M, N = ensemble.shape
    if M < 2:
        return ensemble.copy()
        
    # 1. Compute ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean  # Shape (M, N)
    
    # 2. Compute ensemble state covariance (N x N)
    # P_state = X.T @ X / (M - 1)
    P_state = (X.T @ X) / (M - 1)
    
    # 3. Localization
    # Use a cosine taper with radius L=4
    L = 4
    idx = np.arange(N)
    dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
    
    # Periodic distance (Lorenz-96 is periodic)
    dist = np.minimum(dist, N - dist)
    
    # Taper function
    W = np.zeros((N, N))
    mask = dist <= L
    W[mask] = 1 - 0.5 * (1 - np.cos(np.pi * dist[mask] / L))
    
    # Apply localization to state covariance
    P_loc = P_state * W
    
    # 4. Observation Operator H
    H = np.zeros((len(observed_indices), N))
    for i, obs_idx in enumerate(observed_indices):
        H[i, obs_idx] = 1.0
        
    # 5. Observation Error Covariance R
    R = observation_variance * np.eye(len(observed_indices))
    
    # 6. Kalman Gain
    # S = H P_loc H^T + R
    HP = H @ P_loc  # (O, N) @ (N, N) -> (O, N)
    S = HP @ H.T + R  # (O, N) @ (N, O) -> (O, O)
    
    # Use solve for stability instead of inv
    # K = P_loc H^T S^-1
    # K = (P_loc @ H.T) @ np.linalg.inv(S)
    # Better: solve S^T K^T = H P_loc^T  =>  K = (S^-1 H P_loc)^T ?
    # K (N x O) = P_loc (N x N) @ H.T (N x O) @ S^-1 (O x O)
    # Let A = P_loc @ H.T  (N x O)
    # K = A @ inv(S)
    A = P_loc @ H.T
    # Solve S K^T = A^T  => K = (S^-1 A^T)^T = A S^-1
    # np.linalg.solve(S.T, A.T).T is equivalent to A @ inv(S)
    K = np.linalg.solve(S.T, A.T).T
    
    # 7. Update Mean
    innovation = observations - (H @ x_mean)  # (O,)
    x_mean_new = x_mean + K @ innovation  # (N,) + (N, O) @ (O,) -> (N,)
    
    # 8. Update Anomalies
    # X_new = X @ (I - H^T K)
    # H.T (N, O) @ K (N, O) ? No. K is (N, O). H.T is (N, O).
    # The operator is (I - K H). K (N, O) @ H (O, N) -> (N, N).
    # X is (M, N). We need X_new (M, N).
    # X_new = X @ (I - H.T @ K) ?
    # Let's check the standard additive EnKF anomaly update:
    # x_i' = x_i + K (y - H x_i)
    # x_i' = x_i + K y - K H x_i
    # x_i' = (I - K H) x_i + K y
    # For anomalies: x_i - mean.
    # X' = (I - K H) X
    # Here (I - K H) is (N, N). X is (M, N).
    # (N, N) @ (M, N) is invalid.
    # We must transpose X to (N, M) for the multiplication, then transpose back?
    # Or use the form: X_new = X - X @ H.T @ K ?
    # Let's derive:
    # x_a = x_f + K (y - H x_f)
    # x_a - x_a_mean = (x_f - x_f_mean) + K (y - H (x_f - x_f_mean + x_f_mean)) - K(y - H x_f_mean) ... this is messy.
    # Standard result: X_a = (I - K H) X_f.
    # Since X_f is (M, N), we compute:
    # X_a = X_f - X_f @ H.T @ K ?
    # (M, N) @ (N, O) @ (O, N) ? No, K is (N, O).
    # H.T is (N, O). K is (N, O). H.T @ K is (N, N) ?? No.
    # H.T (N, O) @ K (N, O) is invalid.
    # We need (I - K H). K (N, O) @ H (O, N) -> (N, N).
    # Let M_op = I - K @ H  (N, N)
    # X_a = M_op @ X_f  -> (N, N) @ (M, N) Invalid.
    # X_a = X_f @ M_op.T ?
    # X_f (M, N) @ M_op.T (N, N) -> (M, N). Valid.
    # Is M_op.T = I - H.T @ K.T?
    # M_op = I - K H. M_op.T = I - H.T K.T.
    # So X_new = X @ (I - H.T @ K.T)
    # Let's verify dimensions:
    # H.T (N, O)
    # K.T (O, N)
    # H.T @ K.T -> (N, N)
    # I - ... -> (N, N)
    # X @ ... -> (M, N) @ (N, N) -> (M, N). Correct.
    
    I = np.eye(N)
    # Compute H.T @ K.T
    # H.T is (N, O), K.T is (O, N)
    HKT = H.T @ K.T
    X_new = X @ (I - HKT)
    
    # 9. Variance Preservation (Optional but good for stability)
    # Calculate spread reduction
    # Trace of P_loc before
    tr_old = np.trace(P_loc)
    # Trace of P_new
    P_new = (X_new.T @ X_new) / (M - 1)
    tr_new = np.trace(P_new)
    
    if tr_new > 1e-10:
        alpha = np.sqrt(tr_old / tr_new)
    else:
        alpha = 1.0
        
    # Clip inflation
    alpha = np.clip(alpha, 0.8, 1.2)
    
    X_final = alpha * X_new
    
    # 10. Reconstruct Ensemble
    result = x_mean_new + X_final
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        
    # Clip to [-100, 100]
    result = np.clip(result, -100, 100)
    
    return result
