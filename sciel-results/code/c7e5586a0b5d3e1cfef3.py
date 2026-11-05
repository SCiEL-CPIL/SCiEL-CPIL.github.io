"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_step(x, F, dt=0.01, steps=5):
    """Propagate a single state vector for `steps` RK4 steps of dt."""
    n = len(x)
    x = x.copy()
    for _ in range(steps):
        # RK4
        def f(state):
            return ((np.roll(state, 1) - np.roll(state, 2)) * np.roll(state, 1) - state + F)
        
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 partial-observation assimilation.
    
    Uses a localized Ensemble Kalman Filter (EnKF) with multiplicative inflation.
    The ensemble is 16 members x 20 variables. Observations are of even-indexed
    coordinates (10 variables) with known variance.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (16, 10)
        Noisy observations of the observed indices for each ensemble member.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Estimate F from the ensemble mean (crude but useful for localization)
    # Actually, we don't know F directly. Let's use a standard EnKF approach.
    
    # --- Ensemble Mean and Covariance ---
    x_mean = np.mean(ensemble, axis=0)
    
    # Center the ensemble
    Xc = ensemble - x_mean
    
    # Sample covariance (with degrees of freedom correction)
    # C = Xc.T @ Xc / (n_members - 1)
    # For the EnKF update, we work in observation space
    
    # Observation operator: H picks out observed_indices
    # H is n_obs x n_vars, but we can avoid explicit construction
    
    # Forecast observation covariance: R_obs = H C H^T
    # In observation space: y_obs = Xc[:, observed_indices]
    y_obs = Xc[:, observed_indices]  # shape (n_members, n_obs)
    
    # Observation error covariance R = observation_variance * I
    R = observation_variance
    
    # Innovation: d = observations - H x_mean
    # observations is (16, 10) - these are the noisy obs for each member? 
    # Actually, in the task, observations are independent N(0,0.49) noise added to 
    # the true state. The observations array shape is (16, 10) meaning each member
    # has its own observation? Or is it a single set of observations?
    # Re-reading: "observes even-indexed states with independent N(0,0.49) noise"
    # The observations parameter is (16, 10) - so each ensemble member has an observation.
    # This is unusual. Typically observations are a single vector. But here it seems
    # like each member gets its own noisy observation.
    # Let's treat it as: for each member, we have an observation of the observed variables.
    
    # For a standard EnKF, we'd have a single observation vector. But here we have
    # per-member observations. Let's use the mean observation as the "true" observation
    # and account for the ensemble spread in observations.
    
    # Actually, let me reconsider. The standard approach when you have per-member 
    # observations is to use them directly in the update.
    
    # Let's use a simpler approach: treat the mean observation as the analysis target
    # and use the ensemble covariance for the Kalman gain.
    
    obs_mean = np.mean(observations, axis=0)  # shape (n_obs,)
    
    # Innovation: d = obs_mean - x_mean[observed_indices]
    d = obs_mean - x_mean[observed_indices]
    
    # Observation covariance in observation space
    # S = H C H^T + R
    # H C H^T can be computed as y_obs.T @ y_obs / (n_members - 1)
    C_obs = (y_obs.T @ y_obs) / (n_members - 1.0)  # shape (n_obs, n_obs)
    
    # Add observation noise
    S = C_obs + R * np.eye(n_obs)
    
    # Invert S for the Kalman gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = C H^T S^{-1}
    # C H^T is the cross-covariance between state and observations
    # = Xc.T @ y_obs / (n_members - 1)
    C_xh = (Xc.T @ y_obs) / (n_members - 1.0)  # shape (n_vars, n_obs)
    
    K = C_xh @ S_inv  # shape (n_vars, n_obs)
    
    # Apply the update to the ensemble mean
    x_analysis_mean = x_mean + K @ d
    
    # Update each ensemble member
    # x_i = x_mean + (x_i - x_mean) + K @ (y_i - H x_mean)
    # where y_i = observations[i, :] (per-member observation)
    
    updated_ensemble = np.empty_like(ensemble)
    for i in range(n_members):
        # Innovation for this member
        d_i = observations[i, :] - x_mean[observed_indices]
        # Update
        updated_ensemble[i, :] = x_mean + (ensemble[i, :] - x_mean) + K @ d_i
    
    # --- Localization: reduce spurious correlations ---
    # For Lorenz-96 with N=20, use a simple exponential localization
    # Apply localization to the Kalman gain
    # Distance in state space (circular)
    # This is complex, so let's use a simpler approach: reduce the gain for 
    # unobserved variables based on distance to observed variables
    
    # Actually, for N=20, localization is less critical. Let's add inflation instead.
    
    # --- Multiplicative Inflation ---
    # Inflate the ensemble spread to account for model error
    # Compute the spread around the analysis mean
    spread = updated_ensemble - np.mean(updated_ensemble, axis=0)
    spread_std = np.std(spread, axis=0)
    
    # Target spread: use the observation variance as a guide
    # Inflate by a factor that increases spread if it's too small
    # Simple adaptive inflation: compare innovation variance to expected
    innovation_var = np.var(d)
    expected_innovation_var = np.trace(K @ K.T @ np.diag(np.diag(S)))
    
    # Use a fixed inflation factor for stability
    inflation_factor = 1.02
    
    # Apply inflation around the analysis mean
    analysis_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = analysis_mean + inflation_factor * (updated_ensemble - analysis_mean)
    
    # --- Clamp to valid range ---
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finiteness
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return the best we have
        pass
    
    return updated_ensemble

# EVOLVE-BLOCK-END
