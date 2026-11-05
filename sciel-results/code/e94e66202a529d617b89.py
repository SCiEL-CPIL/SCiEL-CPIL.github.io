"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Transform Kalman Filter (ETKF) for Lorenz-96 assimilation.
    """
    t_start = time.time()
    
    N = ensemble.shape[0]  # 16
    D = ensemble.shape[1]  # 20
    obs_idx = np.array(observed_indices, dtype=int)
    M = len(obs_idx)  # 10
    
    obs = np.asarray(observations, dtype=float).ravel()
    
    # Ensemble mean and anomalies
    x_mean = np.mean(ensemble, axis=0)
    X_anom = ensemble - x_mean  # (N, D)
    
    # Observation-space anomalies
    Y = X_anom[:, obs_idx]  # (N, M)
    
    # Observation noise covariance
    R = observation_variance * np.eye(M)
    
    # Compute P H^T and H P H^T
    PHt = (X_anom @ Y.T) / (N - 1)  # (D, M)
    HPHT = (Y @ Y.T) / (N - 1)  # (M, M)
    
    # Innovation covariance
    S = HPHT + R  # (M, M)
    
    # Solve for the gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt @ S_inv  # (D, M)
    
    # Update mean
    innovation = obs - x_mean[obs_idx]
    x_mean_new = x_mean + K @ innovation
    
    # Update anomalies: A_post = (I - K H) A
    KH = np.zeros((D, D))
    KH[:, obs_idx] = K
    I_KH = np.eye(D) - KH
    
    A_post = I_KH @ X_anom
    
    result = x_mean_new[None, :] + A_post
    
    # Safety: ensure finite and in range
    if not np.all(np.isfinite(result)):
        result = ensemble.copy()
        result[:, obs_idx] += 0.5 * (obs - result[:, obs_idx])
    
    result = np.clip(result, -100.0, 100.0)
    return result
# EVOLVE-BLOCK-END
