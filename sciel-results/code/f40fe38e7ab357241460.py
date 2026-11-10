"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    result[mask1] = (-5.0 / 4.0) * d[mask1]**5 / c**5 + 5.0 * d[mask1]**4 / c**4 - 10.0 * d[mask1]**3 / c**3 + 5.0 * d[mask1]
    # 0.5c < d <= 1.5c
    result[mask2] = (1.0 / 12.0) * d[mask2]**5 / c**5 - (5.0 / 4.0) * d[mask2]**4 / c**4 + (5.0 / 2.0) * d[mask2]**3 / c**3 - 5.0 * d[mask2]**2 / c**2 + (25.0 / 12.0) * d[mask2] / c + 1.0 / 2.0
    # 1.5c < d <= 2.5c
    result[mask3] = (-1.0 / 16.0) * d[mask3]**5 / c**5 + (1.0 / 4.0) * d[mask3]**4 / c**4 - (15.0 / 8.0) * d[mask3]**3 / c**3 + (5.0 / 2.0) * d[mask3]**2 / c**2 - (25.0 / 12.0) * d[mask3] / c + 105.0 / 48.0
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Ensemble covariance (with 1/(M-1) bias correction)
    cov = (anomalies.T @ anomalies) / (M - 1)
    
    # Observation operator: select observed indices
    # H cov H^T + R
    HcovHT = cov[np.ix_(observed_indices, observed_indices)]
    R = observation_variance * np.eye(K)
    
    # Innovation covariance
    S = HcovHT + R
    
    # Gain matrix: K = cov H^T S^{-1}
    # We need cov[:, observed_indices] @ S^{-1}
    cov_H = cov[:, observed_indices]  # (N, K)
    
    # Solve for gain: K = cov_H @ inv(S)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K_gain = cov_H @ S_inv  # (N, K)
    
    # Innovation: y - H x
    y = observations
    Hx = mean[observed_indices]
    innovation = y - Hx
    
    # Update mean
    new_mean = mean + K_gain @ innovation
    
    # Update ensemble
    # Use deterministic update to avoid spurious correlations
    # x_a = x_f + K (y - H x_f)
    result = ensemble.copy()
    for i in range(M):
        Hxi = ensemble[i, observed_indices]
        result[i] = ensemble[i] + K_gain @ (y - Hxi)
    
    # Apply localization to the update to reduce spurious long-range correlations
    # This is a simple approach: scale the update based on distance
    # For Lorenz-96, use periodic distance
    loc_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            d = abs(i - j)
            d = min(d, N - d)  # periodic distance
            loc_matrix[i, j] = _gaspari_cohn(d, c=3.0)
    
    # Apply localization to the gain matrix effect
    # Actually, let's apply localization differently: scale the innovation contribution
    # by the localization function
    
    # Alternative: use a simpler localization approach
    # Scale the update for each state variable based on how well it's constrained
    # by nearby observations
    
    # Compute effective observation density for each state
    obs_density = np.zeros(N)
    for idx in observed_indices:
        for i in range(N):
            d = abs(i - idx)
            d = min(d, N - d)
            obs_density[i] += _gaspari_cohn(d, c=3.0)
    
    # Normalize
    obs_density = obs_density / np.max(obs_density) if np.max(obs_density) > 0 else np.ones(N)
    
    # Blend: use full EnKF update where well-observed, less where not
    # Actually, the standard EnKF already handles this through the covariance
    # Let's just return the standard EnKF result
    
    # Add adaptive inflation to prevent filter collapse
    # Compute ensemble spread vs observation error
    spread = np.std(ensemble[:, observed_indices], axis=0)
    obs_err = np.sqrt(observation_variance)
    
    # If spread is too small relative to observation error, inflate
    ratio = np.mean(spread) / obs_err
    if ratio < 1.0:
        inflation = 1.0 + 0.1 * (1.0 - ratio)
        new_mean_centered = result - np.mean(result, axis=0)
        result = np.mean(result, axis=0) + inflation * new_mean_centered
    
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
