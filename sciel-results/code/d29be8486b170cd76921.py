import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Two-Pass Localized EnKF update with spread-preserving inflation.
    
    Strategy:
    1. First Pass: Localized EnKF with narrow length scale (L=2.5) to capture 
       immediate local dynamics and correct spurious correlations tightly.
    2. Second Pass: Localized EnKF with wider length scale (L=4.0) to smooth 
       out remaining artifacts and ensure consistency over a broader range.
    3. Inflation: Apply multiplicative inflation to restore the ensemble spread 
       to the pre-update level, preventing filter collapse in chaotic systems.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array with shape (16, 20).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

    # Store pre-update statistics for inflation
    mean_pre = np.mean(ensemble, axis=0)
    spread_pre = np.std(ensemble, axis=0, ddof=1)

    def run_enkf_pass(ens, obs, obs_idx, R, L):
        """
        Helper to perform a single localized EnKF pass.
        """
        N, D = ens.shape
        n_obs = len(obs_idx)
        
        # Compute Localization Weights
        # dist is (D, n_obs)
        # For periodic domain, distance is min(|i-j|, D-|i-j|)
        i_range = np.arange(D)
        j_range = obs_idx
        # Create meshgrid for distances
        # i_range[:, None] is (D, 1), j_range[None, :] is (1, n_obs)
        dist = np.abs(i_range[:, None] - j_range[None, :])
        dist = np.minimum(dist, D - dist)
        
        # Exponential localization
        loc_weights = np.exp(-dist / L)  # (D, n_obs)
        
        # Compute Anomalies
        mean = np.mean(ens, axis=0)
        anomalies = ens - mean  # (N, D)
        
        # Extract observed anomalies
        anomalies_obs = anomalies[:, obs_idx]  # (N, n_obs)
        
        # Compute Covariances
        # C_oo (n_obs, n_obs)
        C_oo = (anomalies_obs.T @ anomalies_obs) / (N - 1)
        
        # Innovation Covariance S (n_obs, n_obs)
        S = C_oo + R
        
        # C_xo (D, n_obs)
        C_xo = (anomalies.T @ anomalies_obs) / (N - 1)
        
        # Localized C_xo
        C_xo_loc = C_xo * loc_weights
        
        # Kalman Gain K (D, n_obs)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = C_xo_loc @ S_inv
        
        # Update Mean
        innovation_mean = obs - mean[obs_idx]  # (n_obs,)
        mean_new = mean + K @ innovation_mean
        
        # Update Ensemble
        # x_new_i = x_i + K @ (y - H x_i)
        x_obs_ens = ens[:, obs_idx]  # (N, n_obs)
        innovations_ens = obs[None, :] - x_obs_ens  # (N, n_obs)
        
        # (N, n_obs) @ (n_obs, D) -> (N, D)
        update = innovations_ens @ K.T
        new_ens = ens + update
        
        return new_ens

    R = observation_variance * np.eye(n_obs)
    
    # Pass 1: Narrow localization
    ens_pass1 = run_enkf_pass(ensemble, observations, obs_idx, R, L=2.5)
    
    # Time check
    if time.time() > deadline:
        return np.clip(ens_pass1, -100.0, 100.0)
        
    # Pass 2: Wide localization
    ens_pass2 = run_enkf_pass(ens_pass1, observations, obs_idx, R, L=4.0)
    
    # Time check
    if time.time() > deadline:
        return np.clip(ens_pass2, -100.0, 100.0)

    # Inflation: Restore spread to pre-update levels
    # We want the post-update spread to match the pre-update spread.
    # This is a strong assumption but often stabilizes chaotic filters 
    # better than adaptive ratios which can lag.
    mean_post = np.mean(ens_pass2, axis=0)
    spread_post = np.std(ens_pass2, axis=0, ddof=1)
    
    inflation_factors = np.ones(D)
    for d in range(D):
        if spread_post[d] > 1e-6 and spread_pre[d] > 1e-6:
            # Target spread is spread_pre
            # Current spread is spread_post
            # Factor = spread_pre / spread_post
            # Damp slightly to avoid over-correction if pre-update was already too wide
            target_ratio = spread_pre[d] / spread_post[d]
            # Cap inflation to avoid explosion
            inflation_factors[d] = np.clip(target_ratio, 0.8, 1.5)
        else:
            inflation_factors[d] = 1.0
            
    # Apply inflation
    ens_final = ens_pass2 - mean_post
    ens_final *= inflation_factors
    ens_final += mean_post
    
    # Final Clip
    ens_final = np.clip(ens_final, -100.0, 100.0)
    
    return ens_final
