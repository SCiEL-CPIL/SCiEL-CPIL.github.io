import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Nonlinear Ensemble Smoother (EnSR) update using local linear regression.
    
    This approach fits a local linear map from the observed components of the ensemble
    to the unobserved components. This allows for a nonlinear adjustment of the unobserved
    states based on the observed states, capturing nonlinear dependencies better than
    a standard linear EnKF update.
    
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

    # Create a mask for unobserved indices
    unobs_mask = np.ones(D, dtype=bool)
    unobs_mask[obs_idx] = False
    unobs_idx = np.where(unobs_mask)[0]
    n_unobs = len(unobs_idx)
    
    if n_unobs == 0:
        return ensemble

    # 1. Extract observed and unobserved parts of the ensemble
    X_obs = ensemble[:, obs_idx]       # (N, n_obs)
    X_unobs = ensemble[:, unobs_idx]   # (N, n_unobs)
    
    # 2. Center the ensemble for regression
    # We want to find a linear map A such that X_unobs_i ≈ A @ (X_obs_i - mean_obs) + mean_unobs
    # This is equivalent to OLS regression.
    
    mean_obs = np.mean(X_obs, axis=0)
    mean_unobs = np.mean(X_unobs, axis=0)
    
    X_obs_centered = X_obs - mean_obs       # (N, n_obs)
    X_unobs_centered = X_unobs - mean_unobs # (N, n_unobs)
    
    # 3. Solve for the regression coefficients A
    # X_unobs_centered ≈ X_obs_centered @ A.T
    # A.T = pinv(X_obs_centered.T) @ X_unobs_centered.T
    # A = (pinv(X_obs_centered.T) @ X_unobs_centered.T).T
    # Or more stably: A = (X_unobs_centered.T @ X_obs_centered) @ pinv(X_obs_centered.T @ X_obs_centered)
    
    # Let's use pinv for stability
    try:
        # A shape: (n_unobs, n_obs)
        # We solve X_unobs_centered = X_obs_centered @ A.T
        # A.T = np.linalg.lstsq(X_obs_centered, X_unobs_centered, rcond=None)[0]
        # This is more stable than explicit pinv in some cases, but pinv is fine for small N.
        A_T = np.linalg.pinv(X_obs_centered) @ X_unobs_centered
        A = A_T.T
    except np.linalg.LinAlgError:
        # Fallback to zero update if regression fails
        A = np.zeros((n_unobs, n_obs))
        
    # 4. Apply the update
    # The observed part of the new ensemble is set to the observations (plus noise? No, we want to move towards obs).
    # Standard EnSR: 
    # The new observed state for member i is the observation y (same for all? No, that collapses spread).
    # Actually, EnSR usually projects the ensemble onto the observation space.
    # If we assume the observations are the "true" values for the observed components, 
    # and we want to update the unobserved components based on the linear relationship.
    
    # However, simply setting X_obs to y collapses the spread in observed components to 0.
    # We need to maintain spread.
    
    # Alternative EnSR formulation:
    # We want to update the whole state.
    # Let x_i be the state.
    # We have y.
    # We want x_new such that the map from x_new_obs to x_new_unobs is consistent with the regression,
    # and x_new_obs is close to y.
    
    # A common EnSR update:
    # 1. Compute the regression A.
    # 2. For each member, compute the "implied" unobserved state given the observed state of that member.
    #    x_unobs_implied_i = mean_unobs + A @ (x_obs_i - mean_obs)
    # 3. But we also want the observed state to be close to y.
    #    If we just replace x_obs_i with y, the spread is lost.
    #    
    # Let's use the "Ensemble Kalman Smoother" logic which is similar:
    # The update is:
    # x_new_unobs = mean_unobs + A @ (y - mean_obs) + (x_unobs_i - (mean_unobs + A @ (x_obs_i - mean_obs)))
    # This shifts the mean of the unobserved part towards the regression prediction given y,
    # while preserving the ensemble spread (the perturbation term).
    #
    # What about the observed part?
    # We should also update the observed part to be closer to y, but maintain spread.
    # A standard EnKF update on the observed part:
    # x_new_obs = x_obs_i + K_obs @ (y - x_obs_i)
    # Where K_obs is the gain for the observed part.
    # For the observed part, the "regression" of x_obs on x_obs is identity.
    # So the EnSR update for observed part is just the EnKF update.
    #
    # Let's compute the EnKF gain for the observed part to be consistent.
    # C_oo = cov(X_obs)
    # S = C_oo + R
    # K_obs = C_oo @ inv(S)
    
    C_oo = np.cov(X_obs, rowvar=False) # (n_obs, n_obs)
    R = observation_variance * np.eye(n_obs)
    S = C_oo + R
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        
    K_obs = C_oo @ S_inv # (n_obs, n_obs)
    
    # Initialize new ensemble
    new_ensemble = np.zeros_like(ensemble)
    
    # Update observed components
    # x_new_obs_i = x_obs_i + K_obs @ (y - x_obs_i)
    innovations_obs = observations[None, :] - X_obs # (N, n_obs)
    new_ensemble[:, obs_idx] = X_obs + innovations_obs @ K_obs.T
    
    # Update unobserved components
    # x_new_unobs_i = x_unobs_i + A @ (y - x_obs_i)
    # Wait, the regression was: x_unobs - mean_unobs = A @ (x_obs - mean_obs)
    # So the "predicted" unobserved given x_obs is: pred_unobs = mean_unobs + A @ (x_obs - mean_obs)
    # The innovation in unobserved space is: x_unobs_i - pred_unobs_i
    # The update should shift the mean towards the prediction given y.
    # pred_unobs_given_y = mean_unobs + A @ (y - mean_obs)
    #
    # The EnSR update for unobserved:
    # x_new_unobs_i = x_unobs_i + (pred_unobs_given_y - pred_unobs_given_x_obs_i)
    #                = x_unobs_i + A @ (y - mean_obs) - A @ (x_obs_i - mean_obs)
    #                = x_unobs_i + A @ (y - x_obs_i)
    #
    # This looks correct and preserves spread.
    
    innovations_full = observations[None, :] - X_obs # (N, n_obs)
    correction_unobs = innovations_full @ A.T # (N, n_unobs)
    
    new_ensemble[:, unobs_idx] = X_unobs + correction_unobs
    
    # 5. Inflation
    # EnSR can also underestimate spread, especially in the unobserved components if the regression is strong.
    # We should check the spread of the new ensemble vs the old.
    # A simple adaptive inflation:
    # Calculate the ratio of spreads.
    
    spread_old = np.std(ensemble, axis=0)
    spread_new = np.std(new_ensemble, axis=0)
    
    # Avoid division by zero
    spread_new_safe = np.where(spread_new > 1e-6, spread_new, 1e-6)
    
    # We want spread_new to be at least spread_old (or a fraction of it).
    # Let's target spread_new >= 0.9 * spread_old
    # inflation_factor = np.maximum(1.0, 0.9 * spread_old / spread_new_safe)
    
    # However, if we inflate too much, we introduce noise.
    # Let's use a conservative inflation.
    # Only inflate if spread has collapsed significantly.
    inflation_factor = np.ones(D)
    for d in range(D):
        if spread_new[d] < 0.5 * spread_old[d]:
            inflation_factor[d] = spread_old[d] / spread_new[d]
    
    mean_new = np.mean(new_ensemble, axis=0)
    new_ensemble = new_ensemble - mean_new
    new_ensemble *= inflation_factor
    new_ensemble += mean_new
    
    # 6. Clip to bounds
    new_ensemble = np.clip(new_ensemble, -100.0, 100.0)
    
    # Time check
    if time.time() > deadline:
        return new_ensemble
    
    return new_ensemble
