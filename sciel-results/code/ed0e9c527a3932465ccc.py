import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with localization and multiplicative inflation.
    Reconstructs the full 20-variable state from noisy observations of 10 coordinates.
    """
    # Convert to float64 for numerical stability
    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)
    
    M, N = X_b.shape  # M=16 ensemble members, N=20 state variables
    
    # Compute background statistics
    mean_b = np.mean(X_b, axis=0)
    X_anom = X_b - mean_b  # M x N
    
    # Compute background covariance (N x N)
    P_b = (X_anom.T @ X_anom) / (M - 1)
    
    # Observation operator: extract observed components
    # H is implicitly the identity on observed indices
    # P_byy = P_b[obs_idx][:, obs_idx] (L x L where L = len(obs_idx))
    P_byy = P_b[np.ix_(obs_idx, obs_idx)]
    
    # Innovation covariance: S = H P_b H^T + R
    L = len(obs_idx)
    S = P_byy + R * np.eye(L)
    
    # Compute Kalman gain: K = P_b H^T S^{-1}
    # P_b H^T is N x L (columns of P_b corresponding to observed indices)
    P_bH = P_b[:, obs_idx]  # N x L
    
    # Solve S^T X = P_bH^T => X = S^{-1} P_bH^T
    # Then K = X^T
    # Use solve for stability
    try:
        S_inv_PbH = np.linalg.solve(S.T, P_bH.T)  # L x N
        K = S_inv_PbH.T  # N x L
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        S_inv = np.linalg.pinv(S)
        K = P_bH @ S_inv  # N x L
    
    # Innovation: y_obs - H mean_b
    innovation = y_obs - mean_b[obs_idx]  # L
    
    # Analysis mean: mean_a = mean_b + K @ innovation
    mean_a = mean_b + K @ innovation
    
    # Analysis ensemble: X_a = X_b + K @ (y_obs - H X_b)
    # H X_b for each member: X_b[:, obs_idx]
    # Innovation for each member: y_obs - X_b[:, obs_idx]
    X_obs_members = X_b[:, obs_idx]  # M x L
    member_innovations = y_obs[None, :] - X_obs_members  # M x L
    X_a = X_b + member_innovations @ K.T  # M x N
    
    # Multiplicative inflation to prevent filter collapse
    # Estimate spread reduction and inflate
    spread_b = np.std(X_b, axis=0)  # N
    spread_a = np.std(X_a, axis=0)  # N
    
    # Target spread: use a heuristic based on observation noise and typical state scale
    # For Lorenz-96 with F=6-10, typical std is around 10-15
    # Inflate if analysis spread is too small relative to background
    inflation_factor = 1.0
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            # If spread collapsed significantly, inflate
            if ratio < 0.9:
                inflation_factor = max(inflation_factor, 1.05)
    
    # Apply inflation around the analysis mean
    if inflation_factor > 1.0:
        X_a = mean_a + inflation_factor * (X_a - mean_a)
    
    # Ensure finite values and clip to [-100, 100]
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)
    
    return X_a
# EVOLVE-BLOCK-END
