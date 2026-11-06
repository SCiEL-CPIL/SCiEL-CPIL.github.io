"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

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
    deadline = start_time + 115.0

    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices, dtype=int)

    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()

    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=100.0, neginf=-100.0)

    # --- Localization matrix ---
    # With obs at every 2nd index, L ~ 2.0-3.0 works well
    L_loc = 2.2
    loc_mat = np.ones((D, M))
    for m_idx, obs_i in enumerate(obs_idx):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc_mat[d, m_idx] = np.exp(-0.5 * (dist / L_loc) ** 2)

    # --- Ensemble mean and anomalies ---
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # --- Build augmented ensemble for better covariance estimation ---
    # Add perturbed copies to get ~4x the ensemble size for statistics
    # Use deterministic perturbations based on member index
    n_aug = 4  # number of augmented copies
    aug_anomalies = np.zeros((N * n_aug, D))
    for k in range(n_aug):
        # Small deterministic perturbation scaled by ensemble spread
        spread = np.std(ensemble, axis=0, ddof=1)
        spread = np.maximum(spread, 0.1)  # avoid zero
        # Perturb in different directions
        if k == 0:
            aug_anomalies[k*N:(k+1)*N] = anomalies
        elif k == 1:
            # Perturb by +0.5 * spread in a fixed pattern
            pert = np.zeros((N, D))
            for i in range(N):
                for d in range(D):
                    pert[i, d] = 0.5 * spread[d] * (1.0 if (i + d) % 2 == 0 else -1.0)
            aug_anomalies[k*N:(k+1)*N] = anomalies + pert
        elif k == 2:
            pert = np.zeros((N, D))
            for i in range(N):
                for d in range(D):
                    pert[i, d] = 0.5 * spread[d] * (-1.0 if (i + d) % 2 == 0 else 1.0)
            aug_anomalies[k*N:(k+1)*N] = anomalies + pert
        else:
            # Perturb by 0.3 * spread in another pattern
            pert = np.zeros((N, D))
            for i in range(N):
                for d in range(D):
                    pert[i, d] = 0.3 * spread[d] * (1.0 if (i * 3 + d) % 3 == 0 else -1.0)
            aug_anomalies[k*N:(k+1)*N] = anomalies + pert

    N_aug = N * n_aug
    aug_mean = np.mean(aug_anomalies, axis=0)  # should be ~0
    aug_anomalies_centered = aug_anomalies - aug_mean

    # --- Observation-space quantities from augmented ensemble ---
    aug_obs_anom = aug_anomalies_centered[:, obs_idx]  # (N_aug, M)
    P_obs = (aug_obs_anom.T @ aug_obs_anom) / (N_aug - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R

    # Regularize S
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-5:
        S += (1e-5 - min_eig) * np.eye(M)
    else:
        S += 1e-7 * np.eye(M)

    # --- Cross-covariance (state vs obs) from augmented ensemble ---
    PHt = (aug_anomalies_centered.T @ aug_obs_anom) / (N_aug - 1)  # (D, M)
    PHt_loc = PHt * loc_mat  # (D, M)

    # --- Kalman gain ---
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    K = PHt_loc @ S_inv  # (D, M)

    # --- Update each original member ---
    updated_ensemble = np.zeros_like(ensemble)
    for n in range(N):
        innovation = obs_vec - ensemble[n, obs_idx]  # (M,)
        updated_ensemble[n] = ensemble[n] + innovation @ K

    # --- Adaptive inflation / deflation ---
    # Measure spread in observation space before and after
    obs_spread_pre = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    obs_spread_post = np.std(updated_ensemble[:, obs_idx], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)

    mean_spread_post = np.mean(obs_spread_post)
    target_spread = obs_noise_std * 0.85  # slightly below noise to avoid over-trusting

    if mean_spread_post > 1e-6:
        # Scale factor to bring spread closer to target
        scale = target_spread / mean_spread_post
        scale = np.clip(scale, 0.7, 1.5)
    else:
        scale = 1.0

    # Apply inflation/deflation around the ensemble mean
    upd_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble = upd_mean + scale * (updated_ensemble - upd_mean)

    # Final safety checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)

    return updated_ensemble

# EVOLVE-BLOCK-END
