"""Lorenz-96 ensemble assimilation with localized EnKF and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn 5th-order correlation function for localization."""
    d = np.abs(d)
    c = float(c)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    d1 = d[mask1]
    result[mask1] = (-5.0/4.0)*(d1/c)**5 + 5.0*(d1/c)**4 - 10.0*(d1/c)**3 + 5.0*(d1/c) - 0.25
    
    d2 = d[mask2]
    result[mask2] = (1.0/12.0)*(d2/c)**5 - (5.0/4.0)*(d2/c)**4 + (5.0/2.0)*(d2/c)**3 - 5.0*(d2/c)**2 + (25.0/12.0)*(d2/c) + 0.75
    
    d3 = d[mask3]
    result[mask3] = (-1.0/16.0)*(d3/c)**5 + (1.0/4.0)*(d3/c)**4 - (15.0/8.0)*(d3/c)**3 + (5.0/2.0)*(d3/c)**2 - (25.0/12.0)*(d3/c) + 1.0/2.0
    
    return result


def _localization_matrix(n, observed_indices, c=1.5):
    """Build localization matrix for observed variables."""
    m = len(observed_indices)
    L = np.eye(m)
    for i in range(m):
        for j in range(m):
            if i != j:
                L[i, j] = _gaspari_cohn(observed_indices[i] - observed_indices[j], c)
    return L


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter analysis for Lorenz-96.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed variables.
    observed_indices : array-like
        Indices of observed variables (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    N, n = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    m = len(obs_idx)
    
    # Deadlines
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s kill
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # (N, n)
    
    # Extract observed anomalies
    anom_obs = anomalies[:, obs_idx]  # (N, m)
    
    # Build localization matrix
    L = _localization_matrix(n, obs_idx, c=1.5)
    
    # Compute localized ensemble covariance of observations
    # P_oo = (1/(N-1)) * anom_obs @ anom_obs.T * L
    P_oo = (anom_obs @ anom_obs.T) / (N - 1)
    P_oo_loc = P_oo * L
    
    # Observation noise covariance (diagonal)
    R = observation_variance * np.eye(m)
    
    # Innovation: y - H(x_mean)
    innovation = observations - mean[obs_idx]  # (m,)
    
    # Kalman gain: K = P_xo * (P_oo + R)^{-1}
    # P_xo = (1/(N-1)) * anomalies @ anom_obs.T * L
    P_xo = (anomalies @ anom_obs.T) / (N - 1)
    P_xo_loc = P_xo * L  # (n, m)
    
    # Solve for gain
    try:
        # K = P_xo_loc @ inv(P_oo_loc + R)
        S = P_oo_loc + R  # (m, m)
        # Use Cholesky or direct solve
        K = P_xo_loc @ np.linalg.inv(S)  # (n, m)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        K = P_xo_loc @ np.linalg.pinv(S)
    
    # Analysis update
    # x_a = x_f + K @ innovation
    analysis_mean = mean + K @ innovation  # (n,)
    
    # Update anomalies (square-root form to maintain ensemble structure)
    # We use the standard perturbation form:
    # For each member: x_a_i = x_a_mean + (x_f_i - x_f_mean) * (I - K @ H)
    # But we need to be careful. Let's use the standard EnKF update:
    # x_a_i = x_f_i + K @ (y_i - H(x_f_i))
    # where y_i are perturbed observations.
    
    # Generate perturbed observations
    rng = np.random.default_rng(42)
    perturbed_obs = observations + rng.standard_normal((N, m)) * np.sqrt(observation_variance)
    
    # Compute innovations for each member
    Hx_f = ensemble[:, obs_idx]  # (N, m)
    innovations = perturbed_obs - Hx_f  # (N, m)
    
    # Update each member
    updated_ensemble = ensemble + innovations @ K.T  # (N, n)
    
    # Adaptive inflation
    # Compute innovation variance vs expected
    innovation_var = np.var(innovations, axis=0)  # (m,)
    expected_var = np.diag(P_oo_loc + R)  # (m,)
    
    # Ratio of observed to expected innovation variance
    ratio = np.mean(innovation_var / (expected_var + 1e-10))
    
    # If ratio > 1, inflate the ensemble
    if ratio > 1.0:
        # Inflation factor: use a simple adaptive scheme
        # alpha = 1 + 0.1 * (ratio - 1)
        alpha = 1.0 + 0.15 * (ratio - 1.0)
        alpha = np.clip(alpha, 1.0, 1.5)  # Cap inflation
        
        # Inflate anomalies around the analysis mean
        updated_anoms = updated_ensemble - analysis_mean
        updated_ensemble = analysis_mean + alpha * updated_anoms
    
    # Check deadline
    if time.time() > deadline:
        # Return best effort (already computed)
        pass
    
    # Ensure finite values and within bounds
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
