import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with exponential localization and adaptive multiplicative inflation.
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
    
    # Apply exponential localization to the cross-covariance to reduce spurious correlations
    # This is crucial for small ensemble sizes in high-dimensional systems like Lorenz-96
    # The localization function is: rho(d) = exp(-0.5 * (d / L_loc)^2)
    # where d is the distance between variable i and observation j, and L_loc is the correlation length
    
    # Compute distances between all state variables and observed variables
    # For periodic boundary conditions in Lorenz-96
    # State variables are indexed 0..N-1, observations at obs_idx
    # Distance on a ring of length N: min(|i-j|, N-|i-j|)
    
    # Create distance matrix: N x L
    state_idx = np.arange(N)
    obs_idx_arr = np.array(obs_idx)
    
    # Compute periodic distances
    diffs = np.abs(state_idx[:, None] - obs_idx_arr[None, :])
    distances = np.minimum(diffs, N - diffs)  # N x L
    
    # Correlation length: for Lorenz-96, a reasonable value is around 3-5
    # We use a Gaussian localization kernel
    L_loc = 4.0  # Correlation length scale
    loc_matrix = np.exp(-0.5 * (distances / L_loc) ** 2)  # N x L
    
    # Apply localization to the cross-covariance
    P_bH_loc = P_bH * loc_matrix  # N x L
    
    # Solve S^T X = P_bH_loc^T => X = S^{-1} P_bH_loc^T
    # Then K = X^T
    # Use solve for stability
    try:
        S_inv_PbH = np.linalg.solve(S.T, P_bH_loc.T)  # L x N
        K = S_inv_PbH.T  # N x L
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        S_inv = np.linalg.pinv(S)
        K = P_bH_loc @ S_inv  # N x L
    
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
    
    # Adaptive multiplicative inflation to prevent filter collapse
    # Estimate spread reduction and inflate based on the ratio of analysis to background spread
    spread_b = np.std(X_b, axis=0)  # N
    spread_a = np.std(X_a, axis=0)  # N
    
    # Compute a weighted inflation factor based on the spread ratios
    # Target: maintain a minimum spread ratio to avoid overconfidence
    min_ratio = 0.85  # Minimum acceptable spread ratio
    
    inflation_factors = np.ones(N)
    for i in range(N):
        if spread_b[i] > 1e-6:
            ratio = spread_a[i] / spread_b[i]
            if ratio < min_ratio:
                # Inflate to bring the ratio closer to min_ratio
                # inflation_factor = min_ratio / ratio, but cap it to avoid excessive inflation
                inflation_factors[i] = min(1.15, min_ratio / ratio)
    
    # Apply inflation around the analysis mean for each variable
    if np.any(inflation_factors > 1.0):
        X_a = mean_a[None, :] + inflation_factors[None, :] * (X_a - mean_a[None, :])
    
    # Ensure finite values and clip to [-100, 100]
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)
    
    return X_a
# EVOLVE-BLOCK-END
