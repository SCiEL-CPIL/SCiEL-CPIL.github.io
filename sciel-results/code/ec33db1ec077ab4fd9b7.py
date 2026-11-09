import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Performs a Stochastic Ensemble Kalman Filter (SE-KF) update.
    
    This method adds random perturbations to the observations for each ensemble member
    to maintain the ensemble spread and avoid the variance collapse inherent in 
    deterministic square-root filters.
    
    Parameters:
    ensemble: np.ndarray of shape (16, 20). Current ensemble states.
    observations: np.ndarray of shape (10,). Noisy observations of even indices.
    observed_indices: np.ndarray of shape (10,). Indices of observed variables.
    observation_variance: float. Variance of observation noise.
    
    Returns:
    np.ndarray of shape (16, 20). Updated ensemble states.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # 2. Compute localized background covariance
    # Use Gaussian localization with sigma=3.0
    sigma = 3.0
    i_idx = np.arange(D)
    j_idx = np.arange(D)
    dist = np.abs(i_idx[:, None] - j_idx[None, :])
    dist = np.minimum(dist, D - dist)
    R = np.exp(-dist**2 / (2 * sigma**2))
    
    C = (anomalies.T @ anomalies) / (N - 1)
    C_loc = C * R
    C_loc = (C_loc + C_loc.T) / 2.0
    
    # 3. Observation operator H (selection of indices)
    HCH = C_loc[np.ix_(obs_idx, obs_idx)]
    R_obs = observation_variance * np.eye(n_obs)
    S = HCH + R_obs
    
    # 4. Compute Kalman Gain K
    CHt = C_loc[:, obs_idx]
    
    try:
        L_s = np.linalg.cholesky(S)
        Y = np.linalg.solve(L_s, CHt.T)
        X = np.linalg.solve(L_s.T, Y)
        K = X.T  # K is D x n_obs
    except np.linalg.LinAlgError:
        K = CHt @ np.linalg.pinv(S)
        
    # 5. Stochastic Observation Perturbations
    # For each ensemble member, we add a random draw from N(0, R_obs) to the observations.
    # This is the core of the SE-KF. It ensures that the ensemble mean update is unbiased
    # and the ensemble spread is preserved (on average) without explicit inflation.
    
    # Generate random perturbations for all members and all observed variables
    # shape: (N, n_obs)
    # We use standard normal distribution scaled by sqrt(variance)
    # Since R_obs is diagonal with observation_variance, the std is sqrt(observation_variance)
    std_obs = np.sqrt(observation_variance)
    
    # Ensure we have a reproducible or at least valid random stream. 
    # Using np.random.default_rng is preferred for modern numpy, but standard np.random is fine too.
    # To avoid global state issues, we could use a local generator, but for speed and simplicity,
    # we use the global generator. In a real system, we might want to seed this, 
    # but the task doesn't specify a seed for the analyzer itself, only for the evaluator's truth.
    # The randomness here is part of the algorithm's stochastic nature.
    obs_perturbations = np.random.normal(0, std_obs, size=(N, n_obs))
    
    # 6. Compute Innovations for each member
    # y_i = observations + perturbation_i - H x_i
    # H x_i = x_i[obs_idx]
    # x_i is ensemble[i]
    
    # Background values at observed indices for all members
    # shape: (N, n_obs)
    H_x = ensemble[:, obs_idx]
    
    # Observed values (same for all) + perturbations - H_x
    # observations is (n_obs,), broadcast to (N, n_obs)
    innovations = observations[None, :] + obs_perturbations - H_x
    
    # 7. Update each ensemble member
    # x_i_new = x_i + K @ innovation_i
    # K is (D, n_obs)
    # innovations is (N, n_obs)
    # We need to compute K @ innovations.T for each member? 
    # Actually, K @ innovation_i gives a vector of size D.
    # So updates = K @ innovations.T -> shape (D, N)
    # Then we add updates.T to ensemble.
    
    updates = (K @ innovations.T).T  # shape (N, D)
    
    ensemble_new = ensemble + updates
    
    # 8. Reconstruct Ensemble
    # The stochastic update inherently handles the mean and covariance.
    # However, due to the finite ensemble size, the mean might shift slightly.
    # Standard SE-KF does not re-center the mean explicitly in the update step, 
    # but the expectation of the mean is correct.
    # To ensure numerical stability and exact mean consistency if desired, one could re-center,
    # but that would break the stochastic property slightly. 
    # Usually, no re-centering is done in standard SE-KF.
    
    # 9. Clip to valid range
    ensemble_new = np.clip(ensemble_new, -100.0, 100.0)
    
    # Ensure finite
    if not np.all(np.isfinite(ensemble_new)):
        # Fallback to original ensemble if something went wrong
        return ensemble.copy()
        
    return ensemble_new
# EVOLVE-BLOCK-END
