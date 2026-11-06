"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _ensemble_kalman_filter(ensemble, obs_vec, observed_indices, observation_variance):
    """
    Nonlinear ensemble smoother with backward iteration using Lorenz-96 dynamics.
    Combines EKF initialization with iterative refinement via system equations.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # Estimate F from ensemble mean (F is constant, can be inferred)
    ens_mean = np.mean(ensemble, axis=0)
    # For Lorenz-96, F ≈ mean(x) in steady state, but we'll use a robust estimate
    # Actually, F is not directly observable; we'll use the EKF update first
    # then refine with dynamics
    
    # Phase 1: Standard EKF update
    anomalies = ensemble - ens_mean  # (N, D)
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    
    # Localization: minimum-image distance on ring
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    L = 2.2  # tighter localization
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    
    PHt_loc = PHt * loc_weights  # (D, M)
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    Si_innov = innovations @ S_inv.T  # (N, M)
    delta_x = Si_innov @ PHt_loc.T  # (N, D)
    
    updated_ensemble = ensemble + delta_x  # (N, D)
    
    # Phase 2: Nonlinear refinement using Lorenz-96 dynamics
    # Estimate F from the system: in Lorenz-96, F is the forcing parameter
    # We can estimate F by looking at the mean behavior
    # For a rough estimate: F ≈ mean of (x + (x_{i+1} - x_{i-2})*x_{i-1}) over i and ensemble
    x_ip1 = np.roll(updated_ensemble, -1, axis=1)
    x_im1 = np.roll(updated_ensemble, 1, axis=1)
    x_im2 = np.roll(updated_ensemble, 2, axis=1)
    rhs_estimate = (x_ip1 - x_im2) * x_im1 - updated_ensemble
    # F should be approximately the mean of -rhs + x, but in steady state dx/dt ≈ 0
    # So F ≈ mean((x_{i+1} - x_{i-2})*x_{i-1})
    F_est = np.mean((x_ip1 - x_im2) * x_im1)
    
    # Iterative nonlinear refinement: 
    # For each member, we can do a few steps of "backward" adjustment
    # using the fact that if we know x(t), we can estimate what x(t-dt) should have been
    # This is a simplified adjoint-like approach
    
    dt_refine = 0.05  # small step for refinement
    n_refine_steps = 3  # number of refinement iterations
    
    for step in range(n_refine_steps):
        # For each ensemble member, compute the implied "previous state" 
        # that would lead to the current state via the Lorenz-96 dynamics
        # This is an approximation: we use a linearized backward step
        
        x_curr = updated_ensemble  # (N, D)
        
        # Compute Jacobian of Lorenz-96 at current state (approximate)
        # J_ij = d(x_i')/d(x_j)
        # x_i' = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F
        # d(x_i')/d(x_{i+1}) = x_{i-1}
        # d(x_i')/d(x_{i-2}) = -x_{i-1}
        # d(x_i')/d(x_{i-1}) = x_{i+1} - x_{i-2}
        # d(x_i')/d(x_i) = -1
        
        # Instead of full Jacobian, use a simple iterative correction
        # Assume the true state x_true satisfies:
        # x_curr ≈ x_true + dt * f(x_true)
        # We want to find x_true given x_curr
        # Linearize: x_curr ≈ x_true + dt * (f(x_est) + J(x_est)(x_true - x_est))
        # Solve for x_true iteratively
        
        x_est = x_curr.copy()
        for inner_iter in range(2):
            # Compute f(x_est)
            x_ip1 = np.roll(x_est, -1, axis=1)
            x_im1 = np.roll(x_est, 1, axis=1)
            x_im2 = np.roll(x_est, 2, axis=1)
            f_xest = (x_ip1 - x_im2) * x_im1 - x_est + F_est
            
            # Approximate Jacobian action: J(x_est) * delta
            # For simplicity, use a diagonal approximation with small off-diagonal terms
            # The key coupling is between adjacent indices
            # J has nonzeros at positions (i, i+1), (i, i-2), (i, i-1), (i, i)
            
            # Compute the correction: delta = (x_curr - x_est - dt*f_xest) / (I - dt*J)
            # Approximate (I - dt*J)^{-1} using a few iterations
            residual = x_curr - x_est - dt_refine * f_xest
            
            # Apply a simple relaxation: x_new = x_est + alpha * residual
            alpha = 0.5  # relaxation factor
            x_est = x_est + alpha * residual
        
        updated_ensemble = x_est
    
    # Phase 3: Adaptive inflation
    ens_obs_updated = updated_ensemble[:, observed_indices]
    ens_spread_obs = np.std(ens_obs_updated, axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_spread = np.mean(ens_spread_obs)
    
    ratio = mean_spread / obs_noise_std if obs_noise_std > 0 else 1.0
    target_ratio = 1.15
    inflation = 1.0 + 0.12 * (target_ratio - ratio) / max(ratio, 0.1)
    inflation = np.clip(inflation, 0.95, 1.10)
    
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = updated_mean + inflation * (updated_ensemble - updated_mean)
    
    # Safety: clip and handle NaNs
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


def _particle_filter_update(ensemble, obs_vec, observed_indices, observation_variance, F=None):
    """
    Nonlinear particle filter update using importance weights and kernel density estimation.
    Handles non-Gaussian error distributions inherent in chaotic systems.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    ens_mean = np.mean(ensemble, axis=0)
    
    # Compute likelihood weights based on observation match
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    
    # Innovation
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Gaussian likelihood (log-space for stability)
    R_std = np.sqrt(observation_variance)
    log_weights = -0.5 * np.sum((innovations / R_std) ** 2, axis=1)
    
    # Normalize weights
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weights /= weights.sum()
    
    # Check effective sample size
    ess = 1.0 / np.sum(weights ** 2)
    
    # If ESS is too low, use resampling; otherwise use weighted update
    if ess < N * 0.5:
        # Systematic resampling
        r = np.random.uniform(0, 1.0 / N, 1)
        cum_weights = np.cumsum(weights)
        cum_weights[-1] = 1.0
        indices = np.searchsorted(cum_weights, r + np.arange(N) / N)
        indices = np.clip(indices, 0, N - 1)
        resampled = ensemble[indices]
        
        # After resampling, apply a small Gaussian perturbation to avoid degeneracy
        # Use ensemble spread to set perturbation scale
        ens_spread = np.std(ensemble, axis=0)
        perturbation_scale = 0.1 * ens_spread
        resampled = resampled + np.random.randn(N, D) * perturbation_scale
    else:
        # Weighted kernel density estimate for posterior mean
        # Use Gaussian kernel with bandwidth based on ensemble spread
        bandwidth = 0.5 * np.std(ensemble, axis=0)
        bandwidth = np.maximum(bandwidth, 0.01)
        
        # Compute weighted posterior mean
        # For each state variable, compute weighted average with kernel smoothing
        # This is expensive, so use a simplified approach: weighted mean + correction
        weighted_mean = np.sum(weights[:, None] * ensemble, axis=0)
        
        # Compute weighted covariance for correction
        anomalies = ensemble - weighted_mean
        cov_weighted = (weights[:, None] * anomalies).T @ anomalies / (1.0 - np.sum(weights**2))
        
        # Simple correction: pull toward observation using localized gain
        obs_anomalies = anomalies[:, observed_indices]
        P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
        R = observation_variance * np.eye(M)
        S = P_obs + R
        PHt = (anomalies.T @ obs_anomalies) / (N - 1)
        
        # Localization
        d_idx = np.arange(D)
        loc_weights = np.ones((D, M))
        L = 2.5
        for m_idx, obs_i in enumerate(observed_indices):
            dist = np.abs(d_idx - obs_i)
            dist = np.minimum(dist, D - dist)
            loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
        
        PHt_loc = PHt * loc_weights
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        # Gain correction
        gain = PHt_loc @ S_inv  # (D, M)
        innovation_mean = obs_vec - np.mean(ensemble[:, observed_indices], axis=0)
        correction = gain @ innovation_mean
        
        # Blend: 70% weighted mean, 30% EKF correction
        resampled = ensemble.copy()
        for i in range(N):
            resampled[i] = 0.7 * weighted_mean + 0.3 * (ensemble[i] + correction)
        
        # Re-center to maintain ensemble spread
        resampled_mean = np.mean(resampled, axis=0)
        original_spread = np.std(ensemble, axis=0)
        new_spread = np.std(resampled, axis=0)
        if np.any(new_spread > 0):
            scale = original_spread / np.maximum(new_spread, 0.01)
            scale = np.clip(scale, 0.8, 1.2)
            resampled = resampled_mean + scale * (resampled - resampled_mean)
    
    # Final safety
    resampled = np.nan_to_num(resampled, nan=0.0, posinf=100.0, neginf=-100.0)
    resampled = np.clip(resampled, -100.0, 100.0)
    
    return resampled


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Primary: Nonlinear ensemble smoother with dynamics-based refinement
    result = _ensemble_kalman_filter(
        ensemble, obs_vec, observed_indices, observation_variance
    )
    
    # Optional: If time permits, do a second pass with different localization
    if time.time() < deadline - 5.0:
        # Try a second EKF pass with slightly different localization to average
        ensemble_for_second = result.copy()
        result2 = _ensemble_kalman_filter(
            ensemble_for_second, obs_vec, observed_indices, observation_variance
        )
        # Blend the two results
        result = 0.6 * result + 0.4 * result2
    
    # Deadline guard
    if time.time() > deadline:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
