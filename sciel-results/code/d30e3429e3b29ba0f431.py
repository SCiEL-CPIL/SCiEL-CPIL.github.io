import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Localized Ensemble Square Root Filter (EnSRF) update with Adaptive Inflation.
    
    This strategy uses the Ensemble Square Root Filter formulation to update the ensemble
    mean and spread without adding random noise, which is more stable for small ensembles.
    It includes:
    1. Gaussian Localization to mitigate spurious correlations.
    2. Adaptive Inflation to maintain ensemble spread consistency with observation noise.
    3. Two-pass update (narrow then wide) for numerical stability in highly nonlinear regimes.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin for 120s limit
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

    # Pre-compute localization weights
    # Gaussian localization: w(d) = exp(-0.5 * (d / L)^2)
    # We will use a moderate localization length.
    # For L=96, typical correlation decay is fast. 
    # Let's use L=5.0 for a reasonable balance between bias and variance.
    L_loc = 5.0
    
    loc_weights = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_weights[i, j] = np.exp(-0.5 * (dist / L_loc) ** 2)
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    # --- Adaptive Inflation Logic ---
    # We estimate the required inflation by comparing the ensemble spread in observed 
    # components to the observation noise.
    # If spread < obs_noise, we inflate. If spread > obs_noise, we deflate (carefully).
    
    mean_prior = np.mean(ensemble, axis=0)
    anomalies_prior = ensemble - mean_prior
    
    # Compute innovation vector
    innov = observations - mean_prior[obs_idx]
    
    # Compute ensemble spread covariance in observation space
    # C_oo = (1/(N-1)) * A_obs^T A_obs
    A_obs = anomalies_prior[:, obs_idx]
    C_oo = (A_obs.T @ A_obs) / (N - 1)
    
    # Check for singularity and compute innovation covariance S = C_oo + R
    S = C_oo + R
    
    # Estimate inflation factor
    # Target: The variance of the innovations should match the trace of S.
    # Simple heuristic: Compare mean squared innovation to trace(S)/n_obs
    innov_var = np.mean(innov ** 2)
    s_var = np.trace(S) / n_obs
    
    # If the actual innovation variance is larger than expected, the ensemble was too tight.
    # We want to inflate so that the new S matches the observed innovation statistics.
    # alpha^2 * C_oo + R ~ Innovation Cov
    # This is complex to solve directly. A simpler adaptive scheme:
    # alpha = sqrt( (innov_var - R_diag_mean) / (C_oo_diag_mean) ) if C_oo_diag_mean > 0
    # But we must be careful not to over-inflate.
    
    # Let's use a robust adaptive inflation:
    # 1. Calculate the ratio of observed innovation variance to predicted innovation variance.
    # 2. Adjust inflation to close the gap.
    
    c_oo_diag_mean = np.mean(np.diag(C_oo))
    r_diag_mean = observation_variance
    
    if c_oo_diag_mean > 1e-6:
        # Predicted innovation variance (without inflation)
        pred_innov_var = c_oo_diag_mean + r_diag_mean
        
        # Observed innovation variance
        obs_innov_var = innov_var
        
        # We want alpha such that alpha^2 * c_oo + r ~ obs_innov_var
        # alpha^2 = (obs_innov_var - r) / c_oo
        target_alpha_sq = (obs_innov_var - r_diag_mean) / c_oo_diag_mean
        
        # Clamp alpha to reasonable bounds [0.8, 2.0]
        # If target is negative or too small, use 1.0 (no inflation)
        # If target is too large, cap it to avoid divergence
        if target_alpha_sq > 0:
            alpha = np.sqrt(target_alpha_sq)
        else:
            alpha = 1.0
            
        alpha = np.clip(alpha, 0.8, 2.0)
    else:
        alpha = 1.0

    # --- EnSRF Update ---
    # We perform the update in two steps to improve stability:
    # 1. Update with a tighter localization (L=3.0)
    # 2. Update with a wider localization (L=6.0)
    # This "narrow then wide" strategy helps resolve local features while maintaining global consistency.
    
    def ensrf_update(ens, obs, obs_idx, R, loc_w, alpha):
        N, D = ens.shape
        n_obs = len(obs_idx)
        
        # Apply inflation
        mean = np.mean(ens, axis=0)
        anomalies = ens - mean
        ens_inf = mean + alpha * anomalies
        
        # Compute covariance terms
        A_obs_inf = ens_inf[:, obs_idx] - mean[obs_idx]
        A_all_inf = ens_inf - mean
        
        C_oo = (A_obs_inf.T @ A_obs_inf) / (N - 1)
        S = C_oo + R
        
        # Compute Kalman Gain
        C_xo = (A_all_inf.T @ A_obs_inf) / (N - 1)
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = C_xo @ S_inv
        
        # Apply Localization
        K_loc = K * loc_w
        
        # Update Mean
        innov = obs - mean[obs_idx]
        mean_post = mean + K_loc @ innov
        
        # Update Ensemble (EnSRF style)
        # We want the posterior ensemble to have mean_post and appropriate covariance.
        # Standard EnSRF update:
        # x_post_i = x_prior_i + K (y - x_prior_i[obs])
        # This is equivalent to the stochastic EnKF but without the random noise term,
        # preserving the ensemble size and mean exactly.
        
        ens_post = np.empty_like(ens_inf)
        for i in range(N):
            x_obs_i = ens_inf[i, obs_idx]
            corr = K_loc @ (obs - x_obs_i)
            ens_post[i] = ens_inf[i] + corr
            
        return ens_post, mean_post

    # Step 1: Narrow localization
    L_narrow = 3.0
    loc_w_narrow = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_w_narrow[i, j] = np.exp(-0.5 * (dist / L_narrow) ** 2)
            
    if time.time() < deadline:
        ens_mid, mean_mid = ensrf_update(ensemble, observations, obs_idx, R, loc_w_narrow, alpha)
    else:
        ens_mid = ensemble
        mean_mid = mean_prior

    # Step 2: Wide localization
    L_wide = 6.0
    loc_w_wide = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_w_wide[i, j] = np.exp(-0.5 * (dist / L_wide) ** 2)
            
    if time.time() < deadline:
        ens_final, mean_final = ensrf_update(ens_mid, observations, obs_idx, R, loc_w_wide, alpha)
    else:
        ens_final = ens_mid

    # Final safety checks
    if not np.all(np.isfinite(ens_final)):
        ens_final = np.nan_to_num(ens_final, nan=0.0, posinf=100.0, neginf=-100.0)
        
    ens_final = np.clip(ens_final, -100.0, 100.0)
    
    return ens_final
