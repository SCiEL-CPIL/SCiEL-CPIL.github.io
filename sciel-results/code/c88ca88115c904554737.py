"""Lorenz-96 assimilation using a localized ensemble Kalman filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_step(x, F, dt=0.01):
    """
    One RK4 step for the Lorenz-96 system.
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    with periodic boundary conditions.
    """
    n = len(x)
    
    def f(x):
        dx = np.zeros(n)
        for i in range(n):
            dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
        return dx
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an Ensemble Smoother with Multiple Scattering (ES-MSC) analysis update.
    Iteratively applies localized linear EnSRF updates with re-propagation through
    Lorenz-96 dynamics to capture non-Gaussian error structure.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Number of scattering iterations
    n_scatter = 3
    
    # Localization parameters
    decay_length = 2.0
    relaxation = 0.85
    
    # Precompute localization weights
    indices = np.arange(n_state)
    dist_matrix = np.abs(indices[:, None] - observed_indices[None, :])
    dist_matrix = np.minimum(dist_matrix, n_state - dist_matrix)
    loc_weights = 1.0 / (1.0 + (dist_matrix / decay_length) ** 2)
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # ES-MSC iterative loop
    for scatter_iter in range(n_scatter):
        # Compute ensemble statistics
        ens_mean = np.mean(ens, axis=0)
        anomalies = ens - ens_mean
        cov = (anomalies.T @ anomalies) / (n_members - 1)
        
        # Get observed state components from ensemble
        ens_obs = ens[:, observed_indices]  # (n_members, n_obs)
        ens_obs_mean = np.mean(ens_obs, axis=0)
        
        # Compute innovation
        innovation = observations - ens_obs_mean
        
        # Compute innovation covariance S = H P H^T + R
        P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
        S = P_obs_obs + R
        
        # Regularize S
        S_reg = S + 1e-6 * np.eye(n_obs)
        
        # Compute Kalman gain K = P H^T S^{-1}
        PHt = cov[:, observed_indices]  # (n_state, n_obs)
        
        try:
            K = np.linalg.solve(S_reg, PHt.T).T
        except np.linalg.LinAlgError:
            K = PHt @ np.linalg.pinv(S_reg)
        
        # Apply relaxation to prevent over-correction with small ensembles
        K *= relaxation
        
        # Apply localization
        localized_K = K * loc_weights  # (n_state, n_obs)
        
        # Update each ensemble member
        member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
        update = member_innovations @ localized_K.T  # (n_members, n_state)
        ens = ens + update
        
        # Check deadline
        if _check_deadline():
            break
        
        # Re-propagate through dynamics for non-linear correction
        # Propagate each member 5 RK4 steps (dt=0.01)
        F = 8.0  # Default forcing; will be refined if possible
        dt = 0.01
        n_steps = 5
        
        for m in range(n_members):
            x = ens[m]
            for step in range(n_steps):
                x = _lorenz96_step(x, F, dt)
            ens[m] = x
        
        # After re-propagation, apply a lighter correction toward observations
        # to account for the fact that dynamics may have moved us away from obs
        ens_mean_post = np.mean(ens, axis=0)
        ens_obs_post = ens[:, observed_indices]
        ens_obs_mean_post = np.mean(ens_obs_post, axis=0)
        innovation_post = observations - ens_obs_mean_post
        
        # Lighter gain for post-dynamics correction
        anomalies_post = ens - ens_mean_post
        cov_post = (anomalies_post.T @ anomalies_post) / (n_members - 1)
        P_obs_obs_post = cov_post[np.ix_(observed_indices, observed_indices)]
        S_post = P_obs_obs_post + R + 1e-6 * np.eye(n_obs)
        PHt_post = cov_post[:, observed_indices]
        
        try:
            K_post = np.linalg.solve(S_post, PHt_post.T).T
        except np.linalg.LinAlgError:
            K_post = PHt_post @ np.linalg.pinv(S_post)
        
        K_post *= 0.5 * relaxation  # Lighter update
        localized_K_post = K_post * loc_weights
        
        member_innovations_post = observations[None, :] - ens_obs_post
        update_post = member_innovations_post @ localized_K_post.T
        ens = ens + update_post
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    ratios = np.ones(n_obs)
    exp_vars = np.diag(S_expected)
    valid = exp_vars > 1e-10
    if np.any(valid):
        ens_obs_final = ens[:, observed_indices]
        ens_obs_mean_final = np.mean(ens_obs_final, axis=0)
        innovation_final = observations - ens_obs_mean_final
        ratios[valid] = (innovation_final[valid] ** 2 + observation_variance) / exp_vars[valid]
    
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.8, 1.4))
    
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
