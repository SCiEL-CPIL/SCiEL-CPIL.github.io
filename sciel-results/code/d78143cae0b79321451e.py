import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    # x shape: (16, 20)
    # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # Periodic indices
    n = x.shape[1]
    # Use roll for periodic boundaries
    x_im1 = np.roll(x, 1, axis=1)
    x_ip1 = np.roll(x, -1, axis=1)
    x_im2 = np.roll(x, 2, axis=1)
    return (x_ip1 - x_im2) * x_im1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _get_localization_matrix(indices, L=2.0):
    """
    Generate a Gaussian localization matrix for the observed indices.
    The matrix is symmetric, with values exp(-d^2 / (2*L^2)) where d is the
    circular distance on the ring.
    """
    n_obs = len(indices)
    if n_obs == 0:
        return np.array([[1.0]])
    
    # Calculate pairwise circular distances
    # indices are 0..19
    # d(i, j) = min(|i-j|, 20 - |i-j|)
    idx = np.array(indices)
    diff = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(diff, 20 - diff)
    
    # Gaussian kernel
    L2 = L * L
    loc = np.exp(-0.5 * (dist ** 2) / L2)
    
    # Ensure diagonal is 1
    np.fill_diagonal(loc, 1.0)
    return loc

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    EnKF update for Lorenz-96.
    ensemble: (16, 20)
    observations: (10,)
    observed_indices: array of 10 indices
    observation_variance: float
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    N, n = ensemble.shape  # 16, 20
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Estimate F from the current ensemble mean (heuristic)
    # The system is in a steady state on average, so mean(dx/dt) ~ 0
    # => F ~ mean(x_i)
    # A more robust estimate might be derived from the steady state solution, 
    # but for L96, F is often close to the mean value.
    # Actually, for L96, the mean state <x> satisfies <(x_{i+1}-x_{i-2})x_{i-1}> - <x> + F = 0.
    # In a chaotic steady state, <x> is roughly constant. 
    # A common approximation is F ≈ mean(x).
    # Let's check: if F=8, mean(x) is around 8.
    # We'll use the mean of the ensemble as the best estimate for F.
    F_est = np.mean(ensemble)
    
    # 2. Propagate ensemble forward by 5 steps of dt=0.01
    dt = 0.01
    steps = 5
    current_ensemble = ensemble.copy()
    
    for _ in range(steps):
        current_ensemble = _rk4_step(current_ensemble, dt, F_est)
        
        # Safety check
        if time.time() > deadline:
            break
    
    # 3. EnKF Update
    # Anomaly matrices
    ens_mean = np.mean(current_ensemble, axis=0)
    anomalies = current_ensemble - ens_mean
    
    # Observation anomalies
    obs_ens_mean = ens_mean[obs_idx]
    obs_anomalies = observations - obs_ens_mean  # (n_obs,)
    
    # Covariance matrices
    # P_xx (n, n)
    P_xx = (anomalies @ anomalies.T) / (N - 1.0)
    # P_yy (n_obs, n_obs)
    P_yy = (obs_anomalies[:, None] * obs_anomalies[None, :]) / (N - 1.0)
    
    # Localization
    # Apply localization to P_xx and P_yy? 
    # Standard practice: Localize the analysis update.
    # We construct a localized analysis matrix.
    
    # Let's use the standard EnKF update formula:
    # K = P_xy P_yy^-1
    # x_a = x_f + K (y - y_f)
    # For ensemble:
    # x_a_i = x_f_i + (y_i - y_f_i) K^T  (row vector)
    # Or in matrix form:
    # A_a = A_f + (Y - Y_f) K^T  where Y is (n_obs, N)
    
    # We need P_xy (n, n_obs)
    P_xy = (anomalies[:, obs_idx] @ obs_anomalies.T) / (N - 1.0)
    
    # Add observation noise to P_yy
    R = np.eye(n_obs) * observation_variance
    S = P_yy + R
    
    # Localization strategy:
    # Instead of localizing P directly (which can be tricky with inversion),
    # we can use the "local" EnKF approach where we treat each variable independently
    # but account for correlations.
    # However, a simpler and robust method for this specific problem (ring topology)
    # is to apply a localization matrix to the innovation or the gain.
    
    # Let's try a standard global EnKF first, but with a localized P_yy?
    # No, P_yy is already local (obs are sparse).
    # The issue is spurious correlations in P_xy and P_xx.
    
    # A common trick: Localize the analysis by multiplying the gain with a localization matrix?
    # Or, use the "Local Ensemble Transform Kalman Filter" (LETKF) idea.
    
    # Given the small size (20 vars), let's try a standard EnKF with a
    # localization matrix applied to the covariance terms in the gain calculation.
    
    # Let's compute the gain K = P_xy S^-1
    # To avoid explicit inversion, we can solve the system.
    # But N is small.
    
    # Let's implement a localized EnKF by modifying P_xx and P_xy with a
    # localization matrix L.
    # L(i, j) = exp(-d(i, j)^2 / (2 * L_scale^2))
    
    L_scale = 2.0
    L_mat = _get_localization_matrix(np.arange(20), L_scale)
    
    # Apply localization to P_xx
    P_xx_loc = P_xx * L_mat
    
    # For P_xy, we localize the x-index.
    # P_xy[i, k] corresponds to state var i and obs var k.
    # We multiply by L_mat[i, obs_idx[k]]? No, L_mat is 20x20.
    # We need a localization matrix that maps state indices to obs indices.
    # Let's construct L_xy (20, 10)
    # L_xy[i, k] = exp(-d(i, obs_idx[k])^2 / (2 * L_scale^2))
    
    idx_all = np.arange(20)
    diff_xy = np.abs(idx_all[:, None] - obs_idx[None, :])
    dist_xy = np.minimum(diff_xy, 20 - diff_xy)
    L_xy = np.exp(-0.5 * (dist_xy ** 2) / (L_scale ** 2))
    
    P_xy_loc = P_xy * L_xy
    
    # P_yy is already "local" in the sense that it only involves observed vars,
    # but spurious correlations between observed vars can also occur.
    # Let's localize P_yy too.
    L_yy = _get_localization_matrix(obs_idx, L_scale)
    P_yy_loc = P_yy * L_yy
    
    S_loc = P_yy_loc + R
    
    # Compute Gain
    # K = P_xy_loc S_loc^-1
    # Shape: (20, 10)
    
    try:
        # Solve S_loc K^T = P_xy_loc^T  => K = (S_loc^-1 P_xy_loc^T)^T
        # Or just use lstsq or inv
        K = np.linalg.solve(S_loc, P_xy_loc.T).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = np.linalg.pinv(S_loc) @ P_xy_loc.T
        K = K.T  # Wait, pinv(S) @ P_xy.T gives (10, 20). We want (20, 10).
        # Let's re-verify dimensions.
        # P_xy: (20, 10)
        # S: (10, 10)
        # K = P_xy S^-1 : (20, 10)
        # np.linalg.solve(S, P_xy.T) solves S X = P_xy.T -> X = S^-1 P_xy.T (10, 20)
        # X.T = (P_xy S^-1)^T ? No. (S^-1 P_xy^T)^T = P_xy S^-T. Since S is symmetric, S^-T = S^-1.
        # So X.T = P_xy S^-1 = K. Correct.
        
    # Update Ensemble
    # x_a_i = x_f_i + K (y - y_f_i)
    # y - y_f_i is obs_anomalies (10,)
    # Innovation = obs_anomalies
    innovation = observations - obs_ens_mean  # (10,)
    
    # Update each member
    # current_ensemble (16, 20)
    # K (20, 10)
    # K @ innovation (20,)
    # Add to each row
    update_term = K @ innovation
    current_ensemble = current_ensemble + update_term[None, :]
    
    # 4. Adaptive Inflation (Simple version)
    # Check if spread is too small compared to observation error
    # This is a heuristic. A more rigorous approach uses the likelihood.
    # For now, let's add a simple spread maintenance.
    # If the ensemble spread at observed locations is less than the observation std, inflate.
    
    obs_std_ens = np.std(current_ensemble[:, obs_idx], axis=0)
    obs_std_obs = np.sqrt(observation_variance)
    
    # Target spread ratio ~ 1.0
    # If spread is too small, inflate around the mean
    mean_ens = np.mean(current_ensemble, axis=0)
    
    # Calculate a global inflation factor
    # Average ratio of obs std to ens std
    # Avoid division by zero
    safe_std = np.maximum(obs_std_ens, 1e-6)
    ratio = obs_std_obs / safe_std
    # If ratio > 1, spread is too small.
    # Use a conservative inflation
    if np.mean(ratio) > 1.0:
        # Inflate by a factor that brings spread closer to obs std
        # inflation_factor = np.mean(ratio) ** 0.5 ?
        # Let's use a soft clamp
        target_inflation = 1.0 + 0.5 * (np.mean(ratio) - 1.0)
        # Cap inflation
        target_inflation = min(target_inflation, 1.2)
        
        current_ensemble = mean_ens[None, :] + target_inflation * (current_ensemble - mean_ens[None, :])
        
    # Final safety check: clip to [-100, 100]
    current_ensemble = np.clip(current_ensemble, -100.0, 100.0)
    
    return current_ensemble
# EVOLVE-BLOCK-END
