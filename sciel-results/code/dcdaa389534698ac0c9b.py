"""Lorenz-96 ensemble assimilation with SE-Kalman, adaptive inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) analysis update for Lorenz-96.
    Uses random rotation perturbations, adaptive multiplicative inflation, and Gaussian localization.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # --- Adaptive Multiplicative Inflation ---
    ens_centered = ens - ens_mean
    ens_var_obs = np.var(ens[:, obs_idx], axis=0)
    
    # Target spread: balance between observation noise and model uncertainty
    target_spread = np.sqrt(R_val) * 1.4
    current_spread = np.sqrt(np.maximum(ens_var_obs, 1e-10))
    
    ratio = target_spread / np.maximum(current_spread, 1e-10)
    inflation = np.clip(np.mean(ratio), 0.95, 1.18)
    
    # Blend with a base inflation for stability
    base_inflation = 1.02
    inflation = 0.6 * base_inflation + 0.4 * inflation
    inflation = np.clip(inflation, 1.0, 1.15)
    
    ens = ens_mean + inflation * ens_centered
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean
    
    # --- Localization ---
    # Gaussian localization with sigma tuned for Lorenz-96 correlation length
    sigma = 3.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # Shape (20, n_obs)
    
    # --- SE-Kalman Update ---
    # Step 1: Normalize anomalies at observed locations
    A_obs = A[:, obs_idx]  # Shape (n_members, n_obs)
    
    # Compute the square root of the observation covariance
    # S = (1/(M-1)) * A_obs^T @ A_obs + R
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # Shape (n_obs, n_obs)
    S = C_oo + R_val * np.eye(n_obs)
    
    # Compute cross-covariance with localization
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # Shape (n_vars, n_obs)
    C_xo_loc = C_xo * loc_weights
    
    # Compute gain using Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        # K = C_xo_loc @ inv(S)
        X = np.linalg.solve(L.T, C_xo_loc.T)
        Y = np.linalg.solve(L, X)
        K = Y.T  # Shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # Step 2: Generate random rotation matrix for stochastic perturbation
    # Use QR decomposition of a random matrix
    rng = np.random.default_rng()
    random_matrix = rng.standard_normal((n_members, n_members))
    Q, _ = np.linalg.qr(random_matrix)
    
    # Step 3: Perturb observations with random rotation
    # For each member, generate a perturbed observation
    # The perturbation is: obs_perturbed_i = obs + K_obs @ (Q @ z_i)
    # where z_i is a standard normal vector
    
    # Actually, the standard SE-Kalman approach is:
    # 1. For each member i, create a perturbed observation: obs_i = obs + sqrt(R) * u_i
    #    where u_i are columns of a random orthogonal matrix
    # 2. Update each member using the standard Kalman filter with obs_i
    
    # Let's use the standard SE-Kalman formulation:
    # Generate random orthogonal matrix U (n_obs x n_obs)
    random_obs_matrix = rng.standard_normal((n_obs, n_obs))
    U, _ = np.linalg.qr(random_obs_matrix)
    
    # Perturb observations: obs_perturbed = obs + U @ sqrt(R) * ones? 
    # Actually: obs_perturbed_i = obs + sqrt(R) * U[:, i] for member i
    
    sqrt_R = np.sqrt(R_val)
    obs_perturbed = obs[None, :] + sqrt_R * U.T  # Shape (n_obs, n_obs) -> transpose to (n_obs, n_obs)
    # Wait, we need n_members perturbed observations, but we only have n_obs columns in U
    # Since n_members > n_obs, we need to extend U or use a different approach
    
    # Better approach: use the rotation on the anomaly space
    # The SE-Kalman update can be written as:
    # x_i^a = x_i^f + K @ (obs - H @ x_i^f) + (I - K @ H) @ (Q @ e_i)
    # where e_i are standard basis vectors and Q is random orthogonal
    
    # Simpler: use the fact that for SE-Kalman, we can write:
    # x_i^a = (I - K @ H) @ x_i^f + K @ obs + (I - K @ H) @ (Q @ z)
    # where z is a standard normal vector of dimension n_vars
    
    # Let's use a cleaner formulation:
    # 1. Compute the deterministic part: x_det = (I - K @ H) @ x^f + K @ obs
    # 2. Add stochastic perturbation: x_i^a = x_det + (I - K @ H) @ (Q @ e_i)
    
    # H is the observation operator: H @ x = x[obs_idx]
    # So H is a (n_obs x n_vars) matrix with 1s at observed positions
    
    # Build H implicitly
    # (I - K @ H) @ x = x - K @ x[obs_idx]
    
    # Deterministic part for each member:
    # x_det_i = x_i - K @ (x_i[obs_idx] - obs)
    #         = x_i + K @ (obs - x_i[obs_idx])
    
    # Stochastic part:
    # We need to generate n_members random perturbations
    # Use random orthogonal matrix Q of size (n_vars x n_vars)
    
    random_state_matrix = rng.standard_normal((n_vars, n_vars))
    Q_state, _ = np.linalg.qr(random_state_matrix)
    
    # Compute (I - K @ H) @ Q_state
    # (I - K @ H) @ q = q - K @ q[obs_idx]
    K_H_Q = K @ Q_state[obs_idx, :]  # Shape (n_vars, n_vars)
    I_minus_KH_Q = Q_state - K_H_Q  # Shape (n_vars, n_vars)
    
    # For each member i, the update is:
    # x_i^a = x_i + K @ (obs - x_i[obs_idx]) + I_minus_KH_Q[:, i]
    
    # Vectorized computation:
    # K @ (obs - x_i[obs_idx]) for all i:
    innovations = obs[None, :] - ens[:, obs_idx]  # Shape (n_members, n_obs)
    K_innov = innovations @ K.T  # Shape (n_members, n_vars)
    
    # Stochastic perturbation for each member:
    # Use columns of I_minus_KH_Q
    stochastic_part = I_minus_KH_Q.T  # Shape (n_vars, n_vars) -> transpose to get rows as perturbations
    # Actually, we want the i-th member to get the i-th column of I_minus_KH_Q
    # So stochastic_part should be (n_members, n_vars) where row i is I_minus_KH_Q[:, i]
    stochastic_part = I_minus_KH_Q.T  # Shape (n_vars, n_vars), row i is column i of I_minus_KH_Q
    
    # Since n_members = 16 and n_vars = 20, we need to handle the mismatch
    # Use only the first n_members columns
    stochastic_part = stochastic_part[:n_members, :]  # Shape (n_members, n_vars)
    
    # Combine:
    ens = ens + K_innov + stochastic_part
    
    # Recompute mean and check
    ens_mean_new = np.mean(ens, axis=0)
    
    # Clip and check for finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)
    
    if time.time() > deadline:
        return ens.astype(ensemble.dtype)
        
    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
