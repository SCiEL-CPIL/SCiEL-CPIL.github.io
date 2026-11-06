"""Combined EnKF with LETKF-style spread preservation and tuned parameters."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 148.0  # Safety margin before 150s kill

_PARAMS = {
    'alpha': 1.4,        # localization length (between 1.2 and 1.5)
    'beta': 0.9,         # observation error scaling (trust obs slightly more)
    'gamma': 1.015,      # constant multiplicative inflation
    'min_infl': 1.0,     # floor on inflation
    'max_infl': 1.12,    # ceiling on inflation
}

def _compute_circular_dist(i, obs_idx, N):
    """Circular distance between index i and each observed index."""
    d = np.abs(i - obs_idx)
    d = np.minimum(d, N - d)
    return d

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with:
      - Gaussian localization on cross-covariance
      - Tuned observation-error scaling
      - Adaptive inflation with LETKF-inspired spread preservation
      - Cholesky-based solve for numerical stability
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape  # M=16, N=20
    N_obs = len(obs_idx)

    # Load parameters
    alpha = _PARAMS['alpha']
    beta = _PARAMS['beta']
    gamma = _PARAMS['gamma']
    min_infl = _PARAMS['min_infl']
    max_infl = _PARAMS['max_infl']

    # 1. Background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N

    # 2. Background covariance
    P_b = (W_b.T @ W_b) / (M - 1)  # N x N

    # 3. Extract sub-matrices
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # N x N_obs

    # 4. Observation error covariance (scaled)
    R_eff = np.eye(N_obs) * (R_val * beta)

    # 5. Innovation
    H_mean = mean_b[obs_idx]
    d = y - H_mean  # N_obs

    # 6. Localization on P_xo
    loc_matrix = np.ones((N, N_obs), dtype=np.float64)
    for i in range(N):
        dist = _compute_circular_dist(i, obs_idx, N)
        loc_matrix[i, :] = np.exp(-0.5 * (dist / alpha) ** 2)

    P_xo_loc = P_xo * loc_matrix

    # 7. Innovation covariance and Kalman gain
    S = P_oo + R_eff  # N_obs x N_obs

    # Use Cholesky-based solve for numerical stability
    try:
        L = np.linalg.cholesky(S)
        # K = P_xo_loc @ S^{-1}
        # S^{-1} = L^{-T} @ L^{-1}
        # K = P_xo_loc @ L^{-T} @ L^{-1}
        # Let Y = L^{-1} @ P_xo_loc.T  =>  L @ Y.T = P_xo_loc
        # Actually: solve L.T @ Z = P_xo_loc.T, then K.T = solve(L, Z)
        Z = np.linalg.solve(L.T, P_xo_loc.T)  # N_obs x N
        K_T = np.linalg.solve(L, Z)           # N_obs x N
        K = K_T.T                              # N x N_obs
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
            K = P_xo_loc @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = P_xo_loc @ S_inv

    # 8. Update mean
    mean_a = mean_b + K @ d

    # 9. Update perturbations (dual form)
    H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    update_part = K @ H_W_b_T    # N x M
    W_a_T = W_b.T - update_part  # N x M

    # 10. Adaptive inflation with LETKF-inspired spread preservation
    # Compare observed innovation scatter to expected
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var

    # If actual > expected, ensemble is too tight -> inflate
    adaptive_infl = np.sqrt(ratio)
    adaptive_infl = np.clip(adaptive_infl, min_infl, max_infl)

    # Combine constant and adaptive inflation
    total_infl = gamma * adaptive_infl
    total_infl = np.clip(total_infl, min_infl, max_infl * 1.05)

    W_a_T *= total_infl

    # Re-center
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # 11. Construct analysis ensemble
    X_a = mean_a + W_a_T.T  # M x N

    # 12. Safety: clip and handle NaN/Inf
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a
# EVOLVE-BLOCK-END
