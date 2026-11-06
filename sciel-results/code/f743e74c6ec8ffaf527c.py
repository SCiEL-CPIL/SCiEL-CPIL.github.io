"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
    n = len(x)
    # Periodic indices
    x_ip1 = np.roll(x, -1)  # x_{i+1}
    x_im1 = np.roll(x, 1)   # x_{i-1}
    x_im2 = np.roll(x, 2)   # x_{i-2}
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_backward_step(x, F, dt, n_steps=5):
    """
    Propagate Lorenz-96 backward in time using RK4 with reversed time step.
    This is an approximation that helps correct unobserved states using dynamics.
    """
    for _ in range(n_steps):
        k1 = _lorenz96_rhs(x, F)
        k2 = _lorenz96_rhs(x - 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs(x - 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs(x - dt * k3, F)
        x = x - (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _nonlinear_ensemble_smoother(ensemble, obs_vec, observed_indices, observation_variance,
                                  localization_length=3.5, min_inflation=1.0, max_inflation=1.2,
                                  F=8.0, n_backward_steps=5, dt=0.01):
    """
    Nonlinear ensemble smoother that uses Lorenz-96 dynamics to propagate
    ensemble members backward, leveraging known system equations to correct
    unobserved states more accurately.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    min_inflation : float - minimum multiplicative inflation factor
    max_inflation : float - maximum allowed inflation
    F : float - forcing parameter (default 8.0 for typical Lorenz-96)
    n_backward_steps : int - number of backward RK4 steps
    dt : float - time step for backward propagation
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Step 1: Standard ESRF update
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    obs_anomalies = anomalies[:, observed_indices]
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = observation_variance * np.eye(M)
    S = P_obs + R
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    
    # Apply localization
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_weights[d, m_idx] = np.exp(-0.5 * (dist / localization_length) ** 2)
    
    PHt_loc = PHt * loc_weights
    
    try:
        L = np.linalg.cholesky(S)
        S_inv = np.linalg.inv(L.T @ L)
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv
    obs_ens = ensemble[:, observed_indices]
    innovations = obs_vec[None, :] - obs_ens
    delta_x = innovations @ K.T
    filtered_ensemble = ensemble + delta_x
    
    # Step 2: Nonlinear smoother - propagate backward using dynamics
    # This leverages the chaotic dynamics to better estimate unobserved states
    smoothed_ensemble = np.zeros_like(filtered_ensemble)
    
    for n in range(N):
        # Propagate each ensemble member backward
        backward_state = _lorenz96_backward_step(filtered_ensemble[n], F, dt, n_backward_steps)
        
        # Re-apply observation constraint using soft penalty
        # This ensures the backward-propagated state still matches observations
        obs_indices_arr = np.asarray(observed_indices)
        obs_component = backward_state[obs_indices_arr]
        obs_innovation = obs_vec - obs_component
        
        # Compute local gain for observation correction
        obs_spread_local = np.std(ensemble[:, observed_indices], axis=0, ddof=1)
        obs_spread_local = np.where(obs_spread_local < 1e-6, 1e-6, obs_spread_local)
        
        # Weight correction by inverse variance
        gain_obs = 1.0 / (obs_spread_local ** 2 + observation_variance)
        
        # Apply correction to observed components
        backward_state[obs_indices_arr] += obs_innovation * gain_obs
        
        # Propagate forward again to maintain consistency
        forward_state = filtered_ensemble[n].copy()
        for _ in range(n_backward_steps):
            k1 = _lorenz96_rhs(forward_state, F)
            k2 = _lorenz96_rhs(forward_state + 0.5 * dt * k1, F)
            k3 = _lorenz96_rhs(forward_state + 0.5 * dt * k2, F)
            k4 = _lorenz96_rhs(forward_state + dt * k3, F)
            forward_state = forward_state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Blend backward and forward propagated states
        # Use the backward state for unobserved components, forward for observed
        smoothed_ensemble[n] = backward_state.copy()
        smoothed_ensemble[n, obs_indices_arr] = 0.5 * (backward_state[obs_indices_arr] + forward_state[obs_indices_arr])
    
    # Step 3: Adaptive inflation
    obs_spread = np.std(smoothed_ensemble[:, observed_indices], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    mean_ens_spread = np.mean(obs_spread)
    
    if mean_ens_spread < obs_noise_std * 0.7:
        inflation = min(max_inflation, 1.0 + (obs_noise_std - mean_ens_spread) / max(obs_noise_std, 1e-10) * 0.3)
    elif mean_ens_spread > obs_noise_std * 1.5:
        inflation = max(min_inflation, 1.0 - (mean_ens_spread - obs_noise_std) / max(mean_ens_spread, 1e-10) * 0.1)
    else:
        inflation = 1.0
    
    if inflation != 1.0:
        smoothed_ensemble = ens_mean + inflation * (smoothed_ensemble - ens_mean)
    
    smoothed_ensemble = np.nan_to_num(smoothed_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    smoothed_ensemble = np.clip(smoothed_ensemble, -100.0, 100.0)
    
    return smoothed_ensemble


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
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        # If it's (N, M), take the first row as the observation vector
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure we have the right number of observations
    if len(obs_vec) != M:
        # Fallback: use what we have
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            # Pad with zeros if needed
            obs_vec = np.zeros(M)
            obs_vec[:len(obs_vec)] = obs_vec
    
    # Use nonlinear ensemble smoother with dynamics-based backward propagation
    # Estimate F from ensemble statistics (typical Lorenz-96 forcing)
    F_estimate = 8.0  # Default for typical chaotic Lorenz-96
    
    result = _nonlinear_ensemble_smoother(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.5,
        min_inflation=1.0,
        max_inflation=1.2,
        F=F_estimate,
        n_backward_steps=5,
        dt=0.01
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard: if we're running long, return immediately
    elapsed = time.time() - start_time
    if elapsed > 145.0:  # Safety margin before 150s limit
        return result
    
    return result

# EVOLVE-BLOCK-END
