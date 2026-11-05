"""Lorenz-96 assimilation using a Nonlinear Ensemble Smoother with adaptive Levenberg-Marquardt minimization."""
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

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt=0.01):
    """One RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _propagate_ensemble(ens, F, n_steps=5, dt=0.01):
    """Propagate ensemble through Lorenz-96 dynamics."""
    ens_prop = ens.copy()
    for _ in range(n_steps):
        for m in range(len(ens_prop)):
            ens_prop[m] = _rk4_step(ens_prop[m], F, dt)
    return ens_prop

def _compute_localization_weights(n_state, observed_indices, decay_length=2.5):
    """Compute rational localization weights."""
    indices = np.arange(n_state)
    dist_matrix = np.abs(indices[:, None] - observed_indices[None, :])
    dist_matrix = np.minimum(dist_matrix, n_state - dist_matrix)
    weights = 1.0 / (1.0 + (dist_matrix / decay_length) ** 2)
    return weights

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear Ensemble Smoother analysis update with adaptive Levenberg-Marquardt
    minimization of observation misfit.
    
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
    
    # Infer forcing from ensemble mean (typical Lorenz-96 equilibrium ~ F)
    ens_mean = np.mean(ens, axis=0)
    F = ens_mean[0]  # Use first component as proxy for F
    
    # Observation error covariance
    R = np.eye(n_obs) * observation_variance
    
    # Localization weights
    decay_length = 2.5
    loc_weights = _compute_localization_weights(n_state, observed_indices, decay_length)
    
    # Initialize Levenberg-Marquardt parameters
    lambda_lm = 0.1  # Initial damping factor
    best_cost = np.inf
    best_ens = ens.copy()
    
    # Nonlinear smoother iterations
    n_nl_iter = 5
    for nl_iter in range(n_nl_iter):
        # Compute ensemble mean and anomalies
        ens_mean = np.mean(ens, axis=0)
        anomalies = ens - ens_mean
        cov = (anomalies.T @ anomalies) / (n_members - 1)
        
        # Get observed components
        ens_obs = ens[:, observed_indices]
        ens_obs_mean = np.mean(ens_obs, axis=0)
        
        # Innovation
        innovation = observations - ens_obs_mean
        
        # Compute approximate Jacobian of observation operator w.r.t. state
        # Using finite differences on the ensemble
        H = np.zeros((n_obs, n_state))
        for j in range(n_state):
            # Perturb state component j
            pert = 1e-4
            x_plus = ens_mean.copy()
            x_plus[j] += pert
            x_minus = ens_mean.copy()
            x_minus[j] -= pert
            
            # Propagate both
            x_plus_prop = x_plus.copy()
            x_minus_prop = x_minus.copy()
            for _ in range(5):
                x_plus_prop = _rk4_step(x_plus_prop, F, 0.01)
                x_minus_prop = _rk4_step(x_minus_prop, F, 0.01)
            
            # Compute observation difference
            obs_plus = x_plus_prop[observed_indices]
            obs_minus = x_minus_prop[observed_indices]
            H[:, j] = (obs_plus - obs_minus) / (2 * pert)
        
        # Apply localization to Jacobian
        H_loc = H * loc_weights.T  # (n_obs, n_state)
        
        # Compute gain-like matrix using ensemble covariance
        # K_nl = P H^T (H P H^T + R)^{-1}
        PHt = cov @ H_loc.T  # (n_state, n_obs)
        S = H_loc @ cov @ H_loc.T + R
        S_reg = S + 1e-6 * np.eye(n_obs)
        
        try:
            K_nl = np.linalg.solve(S_reg, PHt.T).T  # (n_state, n_obs)
        except np.linalg.LinAlgError:
            K_nl = PHt @ np.linalg.pinv(S_reg)
        
        # Apply localization to gain
        K_nl_loc = K_nl * loc_weights
        
        # Compute nonlinear update for each member
        # Use member-specific innovations
        member_innovations = observations[None, :] - ens_obs  # (n_members, n_obs)
        
        # Adaptive step size based on Levenberg-Marquardt
        update = member_innovations @ K_nl_loc.T  # (n_members, n_state)
        
        # Compute cost function (sum of squared misfits)
        ens_updated = ens + update
        ens_updated_obs = ens_updated[:, observed_indices]
        misfit = observations[None, :] - ens_updated_obs
        cost = np.sum(misfit ** 2) / n_obs
        
        # Levenberg-Marquardt: accept if cost decreases
        if cost < best_cost:
            best_cost = cost
            best_ens = ens_updated.copy()
            lambda_lm = max(lambda_lm * 0.5, 0.01)  # Reduce damping
        else:
            # Reject and increase damping
            lambda_lm = min(lambda_lm * 2.0, 10.0)
            # Try smaller step
            update_small = update * (1.0 / (1.0 + lambda_lm))
            ens_updated_small = ens + update_small
            ens_updated_small_obs = ens_updated_small[:, observed_indices]
            misfit_small = observations[None, :] - ens_updated_small_obs
            cost_small = np.sum(misfit_small ** 2) / n_obs
            
            if cost_small < best_cost:
                best_cost = cost_small
                best_ens = ens_updated_small.copy()
        
        ens = best_ens.copy()
        
        # Check deadline
        if _check_deadline():
            break
    
    # Use best ensemble found
    ens = best_ens.copy()
    
    # Adaptive inflation based on spread matching
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Expected innovation variance after update
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    # Compute inflation factor
    ratios = np.ones(n_obs)
    exp_vars = np.diag(S_expected)
    valid = exp_vars > 1e-10
    if np.any(valid):
        innovation_final = observations - np.mean(ens[:, observed_indices], axis=0)
        ratios[valid] = (innovation_final[valid] ** 2 + observation_variance) / exp_vars[valid]
    
    # Geometric mean for stability
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    
    # Inflation bounds
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.85, 1.35))
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
