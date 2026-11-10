"""Climate dynamics: partial-observation Lorenz-96 assimilation using Local ESRF."""
import numpy as np

# EVOLVE-BLOCK-START

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    def f(y):
        return (np.roll(y, -1) - np.roll(y, 2)) * np.roll(y, 1) - y + F
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _lorenz96_step(x, F, dt, steps):
    """Propagate Lorenz-96 state by RK4 for given number of steps."""
    for _ in range(steps):
        x = _rk4_step(x, F, dt)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Square Root Filter (ESRF) analysis update.
    
    Parameters:
    -----------
    ensemble : (16, 20) array
        Current ensemble states.
    observations : (10,) array
        Noisy observations of even-indexed coordinates.
    observed_indices : array_like
        Indices of observed coordinates (should be even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : (16, 20) array
        Updated ensemble after analysis.
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Extract observed components from ensemble
    Hx = ensemble[:, obs_idx]  # (n_members, n_obs)
    
    # Compute ensemble mean and anomalies
    x_bar = np.mean(ensemble, axis=0)  # (n_vars,)
    Xa = ensemble - x_bar  # (n_members, n_vars)
    
    # Observation mean and anomalies
    y_bar = np.mean(observations)  # scalar
    Y = observations - y_bar  # (n_obs,)
    
    # Compute ensemble covariance for observed space
    # Cyy = H P H^T / (n-1)
    # H P H^T = (1/n) * sum of (Hx_i - Hx_bar)(Hx_i - Hx_bar)^T
    Hx_bar = np.mean(Hx, axis=0)  # (n_obs,)
    Hx_anom = Hx - Hx_bar  # (n_members, n_obs)
    
    # R = observation error covariance (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Innovation: y - Hx_bar
    d = observations - Hx_bar  # (n_obs,)
    
    # Compute S = H P H^T + R
    # H P H^T = (1/(n-1)) * Hx_anom @ Hx_anom.T
    Cyy = (Hx_anom @ Hx_anom.T) / (n_members - 1)  # (n_obs, n_obs)
    S = Cyy + R  # (n_obs, n_obs)
    
    # Compute Kalman gain in observation space: K = P H^T S^{-1}
    # For ESRF, we work in ensemble space.
    # The transformation is: Xa_new = Xa @ T
    # where T is an orthogonal matrix.
    
    # Compute the square root of S
    # Use Cholesky decomposition
    try:
        Ls = np.linalg.cholesky(S)  # S = Ls @ Ls^T
        Ls_inv = np.linalg.inv(Ls)
    except np.linalg.LinAlgError:
        # Fallback to eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-12)
        Ls_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute the matrix that maps ensemble anomalies in observation space
    # to the innovation space
    # Let Z = Hx_anom @ Ls_inv  (n_members, n_obs)
    # This is the "projected" ensemble anomalies in the whitened observation space
    
    # The ESRF transformation:
    # Xa_new = Xa - Xa @ Z^T @ (Z @ Z^T + I)^{-1} @ Z @ Xa^T ... 
    # Actually, the standard ESRF formula is:
    # Let M = Hx_anom @ Ls_inv  (n_members, n_obs)
    # Then T = I - M^T @ (M @ M^T + I)^{-1} @ M
    # But this is for the case where we want to match the covariance exactly.
    
    # More precisely, the ESRF update is:
    # Xa_new = Xa @ (I - M^T @ (M @ M^T + I)^{-1} @ M)
    # where M = Hx_anom @ Ls_inv
    
    M = Hx_anom @ Ls_inv  # (n_members, n_obs)
    
    # Compute M @ M^T
    MMt = M @ M.T  # (n_obs, n_obs)
    
    # Compute (M @ M^T + I)^{-1}
    # Use Cholesky for stability
    A = MMt + np.eye(n_obs)
    try:
        La = np.linalg.cholesky(A)
        A_inv = np.linalg.inv(La.T @ La)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.inv(A)
    
    # Compute T = I - M^T @ A_inv @ M
    # M^T @ A_inv @ M is (n_members, n_members)
    T = np.eye(n_members) - (M.T @ A_inv @ M)
    
    # Apply the transformation to the ensemble anomalies
    Xa_new = Xa @ T  # (n_members, n_vars)
    
    # Compute the new ensemble mean
    # The new mean should be: x_new = x_bar + P H^T S^{-1} (y - Hx_bar)
    # P H^T = (1/(n-1)) * Xa @ Hx_anom^T
    # So P H^T S^{-1} d = (1/(n-1)) * Xa @ Hx_anom^T @ S^{-1} @ d
    
    # Compute S^{-1} @ d
    try:
        Ls = np.linalg.cholesky(S)
        S_inv_d = np.linalg.solve(Ls.T, np.linalg.solve(Ls, d))
    except np.linalg.LinAlgError:
        S_inv_d = np.linalg.solve(S, d)
    
    # Compute the mean update
    # P H^T S^{-1} d = (1/(n-1)) * Xa @ (Hx_anom^T @ S_inv_d)
    PHtSinvd = (Xa @ (Hx_anom.T @ S_inv_d)) / (n_members - 1)  # (n_vars,)
    
    x_new = x_bar + PHtSinvd
    
    # Reconstruct the ensemble
    updated_ensemble = x_new + Xa_new
    
    # Ensure finite values and clip to [-100, 100]
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
