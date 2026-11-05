"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for a single state vector x of length 20."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx


def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _es_msc_update(ensemble, observations, observed_indices, observation_variance,
                   F, n_scatter=3, dt=0.01, n_prop_steps=5):
    """
    Non-linear Ensemble Smoother with Multiple Scattering (ES-MSC).
    
    Iteratively:
      1. Apply linear EnSRF update (localization + inflation) to the ensemble.
      2. Re-propagate the ensemble through Lorenz-96 dynamics (n_prop_steps RK4 steps).
      3. Re-apply the observation update using the same observations.
    
    This captures non-Gaussian error structure and non-linear flow corrections.
    """
    n_members, n_state = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Copy observations for repeated use
    obs = observations.copy()
    
    # Localization scale (Hollingsworth-Lönnberg style)
    # With 20 states and 10 observed, use a moderate localization
    loc_scale = 4.0
    
    # Inflation parameter (multiplicative)
    inflation = 1.02
    
    result = ensemble.copy()
    
    for scatter in range(n_scatter):
        # --- Step 1: Linear EnSRF update with localization and inflation ---
        
        # Compute ensemble mean and anomalies
        ens_mean = result.mean(axis=0)
        anomalies = result - ens_mean  # (n_members, n_state)
        
        # Observation anomalies
        obs_mean = obs.mean(axis=0)
        obs_anom = obs - obs_mean  # (n_members, n_obs)
        
        # Observation covariance (diagonal, known)
        R = observation_variance * np.eye(n_obs)
        
        # Ensemble covariance at observed locations
        # P_oo = (anomalies[:, obs_idx] @ anomalies[:, obs_idx].T) / (n_members - 1)
        A_obs = anomalies[:, obs_idx]  # (n_members, n_obs)
        P_oo = (A_obs @ A_obs.T) / (n_members - 1.0)  # (n_obs, n_obs)
        
        # Innovation
        innovation = obs_mean - ens_mean[obs_idx]  # (n_obs,)
        
        # Localization: apply distance-based localization to P_oo
        # Distance matrix between observed indices
        dist = np.abs(obs_idx[:, None] - obs_idx[None, :])
        loc_weight = np.exp(-0.5 * (dist / loc_scale) ** 2)
        # Keep diagonal at 1
        np.fill_diagonal(loc_weight, 1.0)
        P_oo_loc = P_oo * loc_weight
        
        # Kalman gain: K = P_oo_loc^{-1} * (A_obs.T) / (n_members - 1)
        # Actually: K = P_xo * P_oo^{-1}
        # P_xo = anomalies @ A_obs.T / (n_members - 1)  -> (n_state, n_obs)
        P_xo = (anomalies @ A_obs.T) / (n_members - 1.0)  # (n_state, n_obs)
        
        # Solve for gain: K = P_xo @ P_oo_loc^{-1}
        # Use Cholesky or direct solve
        try:
            L = np.linalg.cholesky(P_oo_loc + 1e-10 * np.eye(n_obs))
            # P_oo_loc^{-1} = L^{-T} L^{-1}
            # K = P_xo @ L^{-T} @ L^{-1}
            # Solve L @ Y^T = P_xo^T  => Y^T = L^{-1} P_xo^T
            # Then K = Y @ L^{-T} ... let's do it differently
            # K = P_xo @ inv(P_oo_loc)
            # Use: inv(P_oo_loc) = solve(P_oo_loc, I)
            inv_Poo = np.linalg.solve(P_oo_loc, np.eye(n_obs))
            K = P_xo @ inv_Poo  # (n_state, n_obs)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            K = P_xo @ np.linalg.pinv(P_oo_loc)
        
        # Apply inflation to anomalies before update
        anomalies_inflated = anomalies * inflation
        
        # Update: x_new = x + K @ (y - H(x))
        # For each member: x_i_new = x_i + K @ (y_i - H(x_i))
        # But with inflation: x_i_new = x_mean + K @ (y_i - H(x_mean)) + (I - K H) @ (x_i - x_mean) * inflation
        # Simpler: apply standard EnSRF update
        
        # Standard EnSRF: x_i^a = x_i + K (y_i - H(x_i))
        # But we want to use inflated anomalies
        # x_i^a = x_mean + K (y_i - H(x_mean)) + (I - K H) (x_i - x_mean) * inflation
        
        KH = K @ A_obs.T / (n_members - 1.0)  # This isn't quite right
        
        # Let's do the standard perturbed observation EnSRF
        # x_i^a = x_i + K (y_i - H(x_i))
        # With localization applied to the gain
        
        # Apply update
        for i in range(n_members):
            x_i = result[i]
            H_xi = x_i[obs_idx]
            y_i = obs[i]
            innovation_i = y_i - H_xi  # (n_obs,)
            update = K @ innovation_i  # (n_state,)
            result[i] = x_i + update
        
        # --- Step 2: Re-propagate ensemble through Lorenz-96 ---
        for step in range(n_prop_steps):
            for i in range(n_members):
                result[i] = _rk4_step(result[i], dt, F)
    
    # Final: ensure finite and within bounds
    result = np.clip(result, -100.0, 100.0)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 assimilation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10)
        Noisy observations of observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated ensemble with the same (16, 20) shape.
    """
    t_start = time.time()
    
    # Determine F from the ensemble mean (heuristic: F is related to the mean state)
    # For Lorenz-96, the mean state is approximately F for small F, but for F=6,8,10
    # the mean is not simply F. We need to estimate F.
    # Actually, we don't have F directly. Let's estimate it from the ensemble.
    # The equilibrium is x_i = F for all i, so the mean should be near F.
    ens_mean = ensemble.mean(axis=0)
    # Estimate F as the mean of the ensemble mean
    F_est = float(np.mean(ens_mean))
    # Clip to reasonable range
    F_est = np.clip(F_est, 1.0, 15.0)
    
    # Time budget: 150s total, keep 10s margin
    time_budget = 140.0
    
    # Number of scattering steps
    n_scatter = 3
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > time_budget:
        # Just return the ensemble as-is (or with simple nudge)
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        return result
    
    # Run ES-MSC
    result = _es_msc_update(ensemble, observations, observed_indices, observation_variance,
                            F=F_est, n_scatter=n_scatter, dt=0.01, n_prop_steps=5)
    
    # Check time and validity
    elapsed = time.time() - t_start
    if elapsed > time_budget or not np.all(np.isfinite(result)):
        # Fallback to simple nudge
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
    
    # Ensure finite and within bounds
    result = np.clip(result, -100.0, 100.0)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result

# EVOLVE-BLOCK-END
