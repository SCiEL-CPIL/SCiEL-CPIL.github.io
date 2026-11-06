import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Hybrid EnKF update with adaptive inflation and localization.
    This approach uses a standard Ensemble Kalman Filter (EnKF) for the mean update
    but adds a deterministic inflation step to restore ensemble spread, which is
    crucial for chaotic systems like Lorenz-96 where linear updates often overestimate
    confidence (underestimate spread). Localization is applied to prevent spurious
    correlations between distant variables.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array with shape (16, 20).
    """
    start_time = time.time()
    deadline = start_time + 110.0

    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

    # 1. Compute Ensemble Mean and Anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # (N, D)

    # 2. Localization Matrix
    # Exponential decay with length scale L=3.5
    # This is critical for Lorenz-96 to avoid spurious correlations
    L = 3.5
    loc_weights = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_weights[i, j] = np.exp(-dist / L)

    # 3. Compute Covariances (with localization)
    # C_xo: State-Observation covariance (D, n_obs)
    # C_oo: Observation-Observation covariance (n_obs, n_obs)
    
    # Extract observed anomalies
    anomalies_obs = anomalies[:, obs_idx]  # (N, n_obs)
    
    # Compute C_oo (n_obs, n_obs)
    C_oo = (anomalies_obs.T @ anomalies_obs) / (N - 1)
    
    # Add observation noise
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R  # Innovation covariance
    
    # Compute C_xo (D, n_obs) with localization
    C_xo = (anomalies.T @ anomalies_obs) / (N - 1)
    C_xo_loc = C_xo * loc_weights  # Apply localization to state-obs covariance

    # 4. Compute Kalman Gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K = C_xo_loc @ S_inv  # (D, n_obs)

    # 5. Compute Innovations and Update
    # Innovation: y - H*x_mean
    innovation = observations - mean[obs_idx]  # (n_obs,)
    
    # Mean Update
    mean_new = mean + K @ innovation
    
    # Ensemble Update (EnKF)
    # x_i_new = x_i + K @ (y - H*x_i)
    # Vectorized: x_new = x + K @ (y - X_obs)
    # Note: We use the original ensemble for the update to maintain consistency
    x_obs_ensemble = ensemble[:, obs_idx]  # (N, n_obs)
    innovations_ensemble = observations[None, :] - x_obs_ensemble  # (N, n_obs)
    
    # Update each member
    new_ensemble = ensemble + (K @ innovations_ensemble.T).T  # (N, D)

    # 6. Adaptive Inflation (Spread Restoration)
    # The EnKF tends to underestimate spread in chaotic systems.
    # We compare the spread of the updated ensemble to the expected spread.
    # A simple heuristic: inflate to maintain the pre-update spread or a target spread.
    # Here, we use a deterministic inflation factor based on the ratio of 
    # pre-update spread to post-update spread in the observed subspace.
    
    # Pre-update spread (std of anomalies)
    spread_pre = np.std(ensemble, axis=0)
    
    # Post-update spread
    new_mean = np.mean(new_ensemble, axis=0)
    new_anomalies = new_ensemble - new_mean
    spread_post = np.std(new_ensemble, axis=0)
    
    # Inflation factor
    # We want to inflate if spread_post < spread_pre
    # Use a conservative factor to avoid over-inflation
    inflation_factor = np.ones(D)
    for d in range(D):
        if spread_post[d] > 1e-6 and spread_pre[d] > 1e-6:
            ratio = spread_pre[d] / spread_post[d]
            # Only inflate if post-update spread is significantly reduced
            if ratio > 1.0:
                # Damp the inflation to be conservative
                inflation_factor[d] = 1.0 + 0.5 * (ratio - 1.0)
        
    # Apply inflation
    new_ensemble = new_ensemble - new_mean
    new_ensemble *= inflation_factor
    new_ensemble += new_mean

    # 7. Clip to bounds
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    # Time check
    if time.time() > deadline:
        return new_ensemble
    
    return new_ensemble
