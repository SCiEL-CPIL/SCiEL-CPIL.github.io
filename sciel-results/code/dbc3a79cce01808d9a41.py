import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization and inflation for Lorenz-96.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Background Statistics
    mean_b = np.mean(ensemble, axis=0)
    deviations = ensemble - mean_b
    
    # 2. Observed Deviations
    obs_devs = deviations[:, observed_indices]
    
    # 3. Background Covariance (Observed Space)
    # P_bb = 1/(N-1) * obs_devs.T @ obs_devs
    P_bb = (obs_devs.T @ obs_devs) / (n_members - 1)
    
    # 4. Localization
    # Gaussian taper with range 2.5
    sigma_loc = 2.5
    # Distance matrix between observed indices
    # obs_indices is an array of indices in the 20-var state
    dist_mat = np.abs(observed_indices[:, None] - observed_indices[None, :])
    loc_mat = np.exp(-dist_mat**2 / (2.0 * sigma_loc**2))
    
    # Apply localization to covariance
    P_loc = P_bb * loc_mat
    
    # 5. Kalman Gain
    # R = obs_variance * I
    R = observation_variance * np.eye(n_obs)
    S = P_loc + R
    
    # K = P_loc @ inv(S)
    # Solve S.T X.T = P_loc.T  => X = P_loc S^-1
    # K = P_loc S^-1
    K = np.linalg.solve(S.T, P_loc.T).T
    
    # 6. Update
    # Innovations: y - x_i_obs
    innovations = observations[None, :] - ensemble[:, observed_indices]
    
    # x_i_a = x_i_b + K @ innovation_i
    # update shape: (N, 20)
    # K shape: (20, 10) ? No.
    # P_bb is (10, 10). K is (20, 10)??
    # Wait.
    # Standard EnKF:
    # x_a = x_b + K (y - H x_b)
    # Here H is a selection matrix.
    # The gain K maps innovations (10,) to state update (20,).
    # So K should be (20, 10).
    # My P_bb is (10, 10).
    # The formula for K in EnKF is:
    # K = P_xb_yb P_yb_yb^-1
    # P_xb_yb is (20, 10).
    # P_yb_yb is (10, 10).
    # I computed P_bb which is P_yb_yb.
    # I need P_xb_yb.
    
    # Let's re-calculate.
    # deviations: (16, 20)
    # obs_devs: (16, 10)
    # P_xb_yb = 1/(N-1) * deviations.T @ obs_devs  -> (20, 10)
    # P_yb_yb = 1/(N-1) * obs_devs.T @ obs_devs    -> (10, 10)
    
    # I already have P_bb (which is P_yb_yb).
    P_xy = (deviations.T @ obs_devs) / (n_members - 1)
    
    # Apply localization to P_xy?
    # Usually, we localize the covariance in observation space (P_yb_yb) or the gain.
    # Localizing P_yb_yb is what I did above.
    # So K = P_xy @ inv(S)
    
    K = np.linalg.solve(S.T, P_xy.T).T
    
    # 7. Apply Update
    update = innovations @ K.T
    result = ensemble + update
    
    # 8. Inflation
    # Inflate around the new mean
    mean_a = np.mean(result, axis=0)
    # Use a conservative inflation factor
    inflation_factor = 1.02
    result = mean_a + inflation_factor * (result - mean_a)
    
    # 9. Safety Check
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        # Fallback to background or nudging if something went wrong
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        
    return result
