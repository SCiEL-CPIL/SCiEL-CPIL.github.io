"""Lorenz-96 assimilation using a localized Ensemble Square Root Filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Square Root Filter analysis update with adaptive inflation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    
    # Compute anomalies
    anomalies = ens - ens_mean
    
    # Compute ensemble covariance of the observed components
    # P_oo = (1/(M-1)) * A_o^T A_o
    A_o = anomalies[:, observed_indices]  # (M, n_obs)
    P_oo = (A_o.T @ A_o) / (n_members - 1)
    
    # Observation error covariance R
    R = np.eye(n_obs) * observation_variance
    
    # Innovation
    innovation = observations - ens_mean[observed_indices]
    
    # Innovation covariance S = P_oo + R
    S = P_oo + R
    
    # Regularize S for numerical stability
    S_reg = S + 1e-6 * np.eye(n_obs)
    
    # Compute the regression coefficients (Kalman Gain)
    # K = P_xo S^{-1}
    # P_xo = (1/(M-1)) * A_x^T A_o
    A_x = anomalies  # (M, n_state)
    P_xo = (A_x.T @ A_o) / (n_members - 1)
    
    try:
        # K shape: (n_state, n_obs)
        K = np.linalg.solve(S_reg.T, P_xo.T).T
    except np.linalg.LinAlgError:
        K = P_xo @ np.linalg.pinv(S_reg)
    
    # Apply relaxation to prevent over-correction
    relaxation = 0.95
    K *= relaxation
    
    # Apply rational localization (Hamill et al.)
    # Tuned decay_length from 2.5 to 3.5 based on previous improvement trend
    decay_length = 3.5
    indices = np.arange(n_state)
    # Compute circular distance between each state variable and each observed variable
    dist_matrix = np.abs(indices[:, None] - observed_indices[None, :])
    dist_matrix = np.minimum(dist_matrix, n_state - dist_matrix)
    weights = 1.0 / (1.0 + (dist_matrix / decay_length) ** 2)
    localized_K = K * weights  # (n_state, n_obs)
    
    # Update each ensemble member
    # x_a = x_f + K (y - H x_f)
    # For each member m:
    # x_a[m] = x_f[m] + localized_K @ (y - x_f[m, observed_indices])
    member_innovations = observations[None, :] - ens[:, observed_indices]  # (n_members, n_obs)
    update = member_innovations @ localized_K.T  # (n_members, n_state)
    ens = ens + update
    
    # Adaptive inflation
    # Recompute mean and covariance after update
    ens_mean_new = np.mean(ens, axis=0)
    anomalies_new = ens - ens_mean_new
    cov_new = (anomalies_new.T @ anomalies_new) / (n_members - 1)
    
    # Expected innovation variance after update
    P_obs_obs_new = cov_new[np.ix_(observed_indices, observed_indices)]
    S_expected = P_obs_obs_new + R
    
    # Compute inflation factor using spread matching
    ratios = np.ones(n_obs)
    exp_vars = np.diag(S_expected)
    valid = exp_vars > 1e-10
    if np.any(valid):
        # Use innovation squared + obs var as a proxy for realized error
        ratios[valid] = (innovation[valid] ** 2 + observation_variance) / exp_vars[valid]
    
    # Use geometric mean for stability
    log_ratios = np.log(np.clip(ratios, 1e-6, 1e6))
    geo_mean_ratio = np.exp(np.mean(log_ratios))
    
    # Dampen inflation to avoid over-correction
    geo_mean_ratio = 0.5 + 0.5 * geo_mean_ratio
    
    # Refined inflation bounds: [0.9, 1.6]
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.9, 1.6))
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
