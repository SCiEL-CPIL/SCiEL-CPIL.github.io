"""Lorenz-96 assimilation using a dynamical ensemble smoother (ES-4DVar) approach."""
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
    """Vectorized Lorenz-96 right-hand side."""
    n = len(x)
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, F, dt=0.01):
    """One RK4 step for the Lorenz-96 system."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _propagate_ensemble(ens, F, n_steps=5, dt=0.01):
    """Propagate each ensemble member forward by n_steps RK4 steps."""
    out = ens.copy()
    for _ in range(n_steps):
        for m in range(out.shape[0]):
            out[m] = _rk4_step(out[m], F, dt)
    return out

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Dynamical ensemble smoother (ES-4DVar) analysis update.
    
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
    
    ens = ensemble.astype(np.float64).copy()
    
    # Infer forcing from ensemble mean equilibrium: for L96, mean ~ F
    # Use a robust estimate: median of ensemble means across components
    ens_mean = np.mean(ens, axis=0)
    F = float(np.median(ens_mean))
    
    # Number of RK4 steps between updates (per task: 5 steps at dt=0.01)
    n_steps = 5
    dt = 0.01
    
    # ES-4DVar: iterative nudge toward observations through dynamics
    # We work in the "analysis window" framework:
    # The ensemble represents the state at time t. Observations are at time t+5*dt.
    # We propagate forward and apply correction.
    
    # Number of inner iterations for the smoother
    n_inner = 3
    
    # Localized observation operator H: extracts observed indices
    # For each ensemble member, propagate forward then observe
    
    # Pre-compute localization weights for observations
    # Use rational localization with decay length ~ 4
    decay_length = 4.0
    loc_weights = np.zeros((n_state, n_obs))
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            loc_weights[i, j] = 1.0 / (1.0 + (d / decay_length) ** 2)
    
    # ES-4DVar iterations
    for it in range(n_inner):
        # Propagate ensemble forward through dynamics
        ens_fwd = _propagate_ensemble(ens, F, n_steps, dt)
        
        # Compute ensemble mean of propagated observations
        ens_obs = ens_fwd[:, observed_indices]  # (n_members, n_obs)
        ens_obs_mean = np.mean(ens_obs, axis=0)
        
        # Innovation
        innovation = observations - ens_obs_mean  # (n_obs,)
        
        # Compute innovation covariance from ensemble
        obs_anomalies = ens_obs - ens_obs_mean
        P_obs_obs = (obs_anomalies.T @ obs_anomalies) / (n_members - 1)
        R = np.eye(n_obs) * observation_variance
        S = P_obs_obs + R + 1e-6 * np.eye(n_obs)
        
        # Compute gain: for each member, correction in observation space
        # K = P_{x,obs} S^{-1}, where P_{x,obs} is covariance between full state and obs
        # In propagated space:
        state_anomalies = ens_fwd - np.mean(ens_fwd, axis=0)
        P_x_obs = (state_anomalies.T @ obs_anomalies) / (n_members - 1)  # (n_state, n_obs)
        
        try:
            K = np.linalg.solve(S, P_x_obs.T).T  # (n_state, n_obs)
        except np.linalg.LinAlgError:
            K = P_x_obs @ np.linalg.pinv(S)
        
        # Apply localization to gain
        K_loc = K * loc_weights  # (n_state, n_obs)
        
        # Compute per-member corrections in forward space
        member_innov = observations[None, :] - ens_obs  # (n_members, n_obs)
        correction_fwd = member_innov @ K_loc.T  # (n_members, n_state)
        
        # Apply correction in forward space, then we need to "pull back" to initial space
        # For simplicity in ES-4DVar, we apply the correction directly to the forward ensemble
        # and then use the mean correction as a nudge to the initial ensemble
        # Better approach: use the correction as a direct update to initial state
        # via the linearized dynamics (Jacobian approximation)
        
        # Simple approach: apply correction proportionally to initial anomalies
        # This is a first-order 4DVar nudge
        init_mean = np.mean(ens, axis=0)
        init_anom = ens - init_mean  # (n_members, n_state)
        
        # Scale correction: use the mean correction direction
        mean_correction = np.mean(correction_fwd, axis=0)  # (n_state,)
        
        # Distribute correction to initial ensemble members proportionally
        # to their anomaly magnitude in the observed directions
        init_obs_anom = init_anom[:, observed_indices]  # (n_members, n_obs)
        # Project correction back using the same localization structure
        # Use a simple proportional nudge
        nudge_scale = 0.7  # damping factor for the nudge
        
        # Apply nudge to initial ensemble
        # Each member gets a fraction of the mean correction weighted by its innovation
        for m in range(n_members):
            # Weight by how much this member's observed anomaly differs from mean
            member_weight = 1.0 + 0.3 * (np.dot(init_obs_anom[m], innovation) / 
                                          (np.dot(innovation, innovation) + 1e-10))
            ens[m] += nudge_scale * member_weight * mean_correction * 0.5
    
    # Final: apply a direct localized EnKF-style update as well for robustness
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    ens_obs = ens[:, observed_indices]
    ens_obs_mean = np.mean(ens_obs, axis=0)
    innovation = observations - ens_obs_mean
    
    P_obs_obs = cov[np.ix_(observed_indices, observed_indices)]
    R = np.eye(n_obs) * observation_variance
    S = P_obs_obs + R + 1e-6 * np.eye(n_obs)
    
    PHt = cov[:, observed_indices]
    try:
        K = np.linalg.solve(S, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S)
    
    # Apply localization
    K_loc = K * loc_weights
    
    # Relaxation
    K_loc *= 0.85
    
    member_innov = observations[None, :] - ens_obs
    update = member_innov @ K_loc.T
    ens = ens + update
    
    # Adaptive inflation
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    ratios = np.ones(n_obs)
    for j in range(n_obs):
        exp_var = S_expected[j, j]
        if exp_var > 1e-10:
            ratios[j] = (innovation[j] ** 2 + observation_variance) / exp_var
    
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.92, 1.35))
    
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip and ensure finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
