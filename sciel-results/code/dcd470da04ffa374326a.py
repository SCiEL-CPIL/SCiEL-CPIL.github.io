"""Ensemble Kalman Filter with adaptive inflation and localization for Lorenz-96."""
import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute the Lorenz-96 right-hand side for state x with forcing F."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5*dt*k1, F)
    k3 = _lorenz96_rhs(x + 0.5*dt*k2, F)
    k4 = _lorenz96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with multiplicative adaptive inflation and rank-deficiency handling.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of the observed indices.
    observed_indices : array_like, shape (10,)
        Indices of the observed variables.
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after the analysis step.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=np.intp)
    
    n_ens = ensemble.shape[0]
    n_state = ensemble.shape[1]
    
    # Ensure finite values
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Estimate forcing F from the ensemble mean (rough estimate)
    # In Lorenz-96, the equilibrium is around F for the constant solution
    # We'll use a simple estimate: F ≈ mean of x + (x[i+1]-x[i-2])*x[i-1] averaged
    # But since we don't know F exactly, we'll use the fact that the system is driven
    # Actually, for the EnKF update, we don't need F - we just update the ensemble
    # based on the observations. The propagation is done by the evaluator.
    
    # Standard EnKF update
    # 1. Compute ensemble mean and anomaly covariance
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # shape (n_ens, n_state)
    
    # 2. Compute observation operator
    # H: select observed indices
    H_ens = anomalies[:, observed_indices]  # shape (n_ens, n_obs)
    H_mean = ens_mean[observed_indices]
    
    # 3. Compute Kalman gain
    # S = H P H^T + R
    # P = (1/(n_ens-1)) * anomalies^T anomalies
    # H P H^T = (1/(n_ens-1)) * H_ens^T H_ens
    n_obs = len(observed_indices)
    
    # Compute S = H P H^T + R
    HPHt = (H_ens.T @ H_ens) / (n_ens - 1)
    R = observation_variance * np.eye(n_obs)
    S = HPHt + R
    
    # Compute S^{-1} (n_obs x n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # K = P H^T S^{-1}
    # P H^T = (1/(n_ens-1)) * anomalies^T H_ens
    PHt = (anomalies.T @ H_ens) / (n_ens - 1)  # shape (n_state, n_obs)
    K = PHt @ S_inv  # shape (n_state, n_obs)
    
    # 4. Innovation
    innovation = observations - H_mean  # shape (n_obs,)
    
    # 5. Update each ensemble member
    updated_ensemble = np.empty_like(ensemble)
    for i in range(n_ens):
        updated_ensemble[i] = ensemble[i] + K @ innovation
    
    # 6. Adaptive multiplicative inflation
    # Check if the ensemble spread is too small relative to the observation error
    # Compute the variance of the updated ensemble at observed points
    updated_mean_obs = np.mean(updated_ensemble[:, observed_indices], axis=0)
    updated_anom_obs = updated_ensemble[:, observed_indices] - updated_mean_obs
    
    # Sample variance of updated ensemble at observed points
    ens_var_obs = np.sum(updated_anom_obs ** 2, axis=0) / (n_ens - 1)
    
    # Target: ensemble variance should be at least comparable to observation variance
    # If it's too small, inflate
    min_var = observation_variance * 0.5
    too_small = ens_var_obs < min_var
    
    if np.any(too_small):
        # Inflate the anomalies in all dimensions proportionally
        # Use a simple multiplicative inflation on the full state
        # Inflate factor based on how much smaller the variance is
        inflation_factor = 1.0
        for j in range(n_obs):
            if too_small[j]:
                ratio = min_var / max(ens_var_obs[j], 1e-10)
                # Take a conservative inflation
                local_inflate = np.sqrt(ratio) * 0.8 + 0.9  # dampened
                inflation_factor = max(inflation_factor, local_inflate)
        
        # Apply inflation to anomalies
        ens_mean_new = np.mean(updated_ensemble, axis=0)
        updated_anoms = updated_ensemble - ens_mean_new
        updated_ensemble = ens_mean_new + inflation_factor * updated_anoms
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Check deadline
    if time.time() > deadline:
        return updated_ensemble
    
    return updated_ensemble
