"""Rewritten program for Lorenz-96 assimilation with localized Ensemble Kalman Filter."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x has shape (N, 20) or (20,)."""
    if x.ndim == 1:
        x_1 = np.roll(x, 1)
        x_m1 = np.roll(x, -1)
        x_m2 = np.roll(x, 2)
        return (x_1 - x_m2) * x_m1 - x + F
    else:
        x_1 = np.roll(x, 1, axis=1)
        x_m1 = np.roll(x, -1, axis=1)
        x_m2 = np.roll(x, 2, axis=1)
        return (x_1 - x_m2) * x_m1 - x + F


def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4 for n_steps."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def _localization_matrix(n_obs, n_state, decay_length):
    """Create a localization matrix for observation space.
    
    Uses a Gaussian decay: w_ij = exp(-d_ij^2 / (2 * decay_length^2))
    where d_ij is the circular distance between observed indices i and j.
    """
    # This is for the observation covariance, but we actually need
    # localization in the state space for the analysis update.
    # We'll handle this differently in the main analyze function.
    pass


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Kalman Filter analysis update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Make a copy to work with
    result = ensemble.copy()
    
    # Estimate forcing parameter F from the ensemble
    # For Lorenz-96, we can estimate F from the mean behavior
    # But we don't have F directly, so we'll use a default or estimate
    # Actually, we can't easily estimate F from just one step, so we'll
    # use the ensemble statistics
    
    # Compute ensemble mean and covariance
    ens_mean = np.mean(result, axis=0)
    anomalies = result - ens_mean
    
    # Compute sample covariance (N-1 normalization)
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error covariance (diagonal)
    R = np.eye(n_obs) * observation_variance
    
    # Localized Ensemble Kalman Filter
    # Use a simple localization: only update states near observed indices
    # with a decay based on distance
    
    # For each observed variable, compute the Kalman gain
    # and apply it with localization
    
    # Observation innovation
    innovation = observations - H @ ens_mean
    
    # Innovation covariance: S = H P H^T + R
    HP = H @ cov
    S = HP @ H.T + R
    
    # Kalman gain: K = P H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = (cov @ H.T) @ S_inv  # Shape: (n_state, n_obs)
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    decay_length = 3.0  # Localization length scale
    
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            # Circular distance
            d = min(abs(i - obs_idx), n_state - abs(i - obs_idx))
            w = np.exp(-d**2 / (2 * decay_length**2))
            localized_K[i, j] = K[i, j] * w
    
    # Update each ensemble member
    # Perturbed observations approach or deterministic update
    # We'll use the deterministic (analysis mean) update
    for m in range(n_members):
        # For each member, adjust based on the innovation
        # The update for each member: x_m^a = x_m^f + K * (y - H x_m^f)
        # But we need to preserve the ensemble spread properly
        
        # Use the ensemble mean innovation for all members (simpler)
        # or perturb observations for each member
        # Let's use the standard ensemble Kalman filter update
        
        # Compute H x_m^f
        Hx_m = H @ result[m]
        
        # Innovation for this member
        innov_m = observations - Hx_m
        
        # Update
        result[m] = result[m] + localized_K @ innov_m
    
    # Adaptive inflation to prevent filter collapse
    # Compute the ratio of ensemble spread to innovation
    # Simple adaptive inflation: inflate if ensemble spread is too small
    
    # Estimate expected innovation variance
    S_expected = HP @ H.T + R
    S_expected_diag = np.diag(S_expected)
    
    # Actual innovation variance
    innov_var = np.var(innovation)
    
    # Inflation factor
    # If the ensemble spread is too small, inflate
    # Target: S_actual ≈ S_expected
    if np.any(S_expected_diag > 0):
        ratio = np.median(innov_var / S_expected_diag)
        # Adjust inflation to match observed to expected
        # alpha = sqrt(expected / actual)
        inflation = np.sqrt(np.clip(ratio, 0.5, 2.0))
    else:
        inflation = 1.0
    
    # Apply inflation around the analysis mean
    new_mean = np.mean(result, axis=0)
    result = new_mean + inflation * (result - new_mean)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
