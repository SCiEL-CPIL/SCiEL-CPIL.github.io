import numpy as np
import time
# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 148.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    if N_obs == 0 or N_obs > N:
        return X_b

    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b

    # Localization: Gaussian kernel
    alpha = 1.5
    loc_matrix = np.ones((N, N_obs), dtype=np.float64)
    for i in range(N):
        # Compute circular distance to each observed index
        dists = np.abs(i - obs_idx)
        dists = np.minimum(dists, N - dists)
        loc_matrix[i, :] = np.exp(-0.5 * (dists / alpha) ** 2)

    # Background Covariance
    P_b = (W_b.T @ W_b) / (M - 1)
    
    # Localized Covariances
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]
    P_xo_loc = P_xo * loc_matrix

    # Innovation Variance
    S = P_oo + np.eye(N_obs) * R_val

    # Gain
    try:
        K = np.linalg.solve(S.T, P_xo_loc.T).T
    except np.linalg.LinAlgError:
        K = P_xo_loc @ np.linalg.pinv(S)

    # Innovation
    innovation = y - mean_b[obs_idx]

    # Update Mean
    mean_a = mean_b + K @ innovation

    # Update Perturbations (Square Root EnKF style to preserve covariance structure better)
    # W_a = W_b - W_b[obs_idx, :] @ K.T
    W_a = W_b - W_b[obs_idx, :] @ K.T

    # Adaptive Inflation
    # Calculate innovation variance per state or globally
    # Simple global adaptive inflation
    # If ensemble spread is too small, inflate
    spread_b = np.sqrt(np.diag(P_b))
    spread_a = np.sqrt(np.diag((W_a.T @ W_a) / (M - 1)))
    
    # Target spread based on innovation magnitude
    # A simple heuristic: if innovation is large relative to obs error, spread might be too small
    # Or just use a fixed inflation factor that is known to work well for L96
    # Knowledge suggests "Spatially adaptive inflation... improves robustness"
    
    # Let's implement a simple spatially adaptive inflation
    # Inflate where the innovation is large
    infl_factor = np.ones(N)
    for i in range(N):
        # Distance to nearest obs
        dists = np.abs(i - obs_idx)
        dists = np.minimum(dists, N - dists)
        min_dist = np.min(dists)
        
        # Base inflation
        base_infl = 1.01
        # Increase inflation if far from observations (less constrained)
        # Or if innovation is large?
        # For L96, a slightly higher inflation helps prevent collapse
        infl_factor[i] = 1.02 + 0.01 * (min_dist / (N/4.0))
        
    W_a = W_a * infl_factor

    # Ensure mean consistency
    W_a -= np.mean(W_a, axis=0, keepdims=True)

    X_a = mean_a + W_a

    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a
# EVOLVE-BLOCK-END
