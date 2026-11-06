"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system using vectorized operations."""
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


def _compute_localization_matrix(D, observed_indices, L):
    """Precompute localization weights for all state-obs pairs."""
    M = len(observed_indices)
    loc = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            loc[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return loc


def _ensemble_analysis(ensemble, obs_vec, observed_indices, observation_variance,
                       localization_length=3.0, max_inflation=2.5, min_inflation=1.0):
    """
    Ensemble analysis with localized KF, innovation-based adaptive inflation.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec).flatten()
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation-space anomalies
    obs_anoms = anomalies[:, observed_indices]  # (N, M)
    
    # Innovation covariance
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R  # (M, M)
    
    # Cross-covariance
    PHt = (anomalies.T @ obs_anoms) / (N - 1)  # (D, M)
    
    # Localization
    loc = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc  # (D, M)
    
    # Invert S
    try:
        L_chol = np.linalg.cholesky(S)
        S_inv = np.linalg.inv(L_chol.T @ L_chol)
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
    
    # Gain
    K = PHt_loc @ S_inv  # (D, M)
    
    # Innovations
    obs_ens = ensemble[:, observed_indices]  # (N, M)
    innovations = obs_vec[None, :] - obs_ens  # (N, M)
    
    # Update
    delta_x = innovations @ K.T  # (N, D)
    updated = ensemble + delta_x
    
    # Innovation-based adaptive inflation
    # Expected innovation variance = trace(S)/M
    expected_innov_var = np.trace(S) / M
    # Actual innovation variance (mean over obs and ensemble)
    actual_innov_var = np.mean(innovations ** 2)
    
    # Inflate if innovations are too large (under-dispersed ensemble)
    if expected_innov_var > 1e-10:
        ratio = actual_innov_var / expected_innov_var
        if ratio > 1.05:
            # Need inflation
            inflation = min(max_inflation, 1.0 + (ratio - 1.0) * 0.5)
        elif ratio < 0.9:
            # Slight deflation
            inflation = max(min_inflation, 1.0 - (1.0 - ratio) * 0.2)
        else:
            inflation = 1.0
    else:
        inflation = 1.0
    
    # Apply inflation about the ensemble mean (use pre-update mean for stability)
    if inflation != 1.0:
        updated = ens_mean + inflation * (updated - ens_mean)
    
    # Safety
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    """
    start_time = time.time()
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0].copy()
    else:
        obs_vec = obs_arr.flatten().copy()
    
    if len(obs_vec) != M:
        if len(obs_vec) > M:
            obs_vec = obs_vec[:M]
        else:
            padded = np.zeros(M)
            padded[:len(obs_vec)] = obs_vec
            obs_vec = padded
    
    # Run analysis
    result = _ensemble_analysis(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=3.0,
        max_inflation=2.5,
        min_inflation=1.0
    )
    
    # Final safety
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        return result
    
    return result

# EVOLVE-BLOCK-END
