"""
Ensemble Square Root Filter (EnSRF) for Lorenz-96 assimilation.
Implements coordinate-wise sequential regression of unobserved variables onto observed ones
using the ensemble covariance, with localization and inflation.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20)."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _rk4_step(x, F, dt, n_steps=1):
    """Propagate ensemble using RK4 for n_steps steps."""
    x = x.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _localize_cov(cov, indices, alpha=1.0):
    """Apply exponential localization to covariance matrix."""
    n = len(indices)
    # Compute distance matrix
    idx = np.array(indices)
    dist = np.abs(idx[:, None] - idx[None, :])
    # Wrap-around distance for periodic domain
    dist = np.minimum(dist, 20 - dist)
    # Exponential decay
    mask = np.exp(-dist / alpha)
    return cov * mask

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using coordinate-wise EnSRF regression.
    
    Parameters:
    -----------
    ensemble : (16, 20) array
        Ensemble of 16 members, 20 variables each
    observations : (16, 10) array
        Noisy observations of even-indexed variables (0, 2, 4, ..., 18)
        One observation per ensemble member
    observed_indices : array-like
        Indices of observed variables (even indices: 0, 2, 4, ..., 18)
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array
    """
    start_time = time.time()
    deadline = 150.0  # seconds
    
    ensemble = np.array(ensemble, dtype=np.float64)
    observations = np.array(observations, dtype=np.float64)
    observed_indices = np.array(observed_indices, dtype=np.int32)
    
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Determine unobserved indices
    all_indices = np.arange(n_vars)
    unobserved_indices = np.setdiff1d(all_indices, observed_indices)
    
    # Compute ensemble mean and centered ensemble
    ens_mean = np.mean(ensemble, axis=0)
    ens_centered = ensemble - ens_mean  # (n_members, n_vars)
    
    # Compute sample covariance (with Bessel correction)
    # cov = ens_centered.T @ ens_centered / (n_members - 1)
    # But we'll work with centered ensemble directly for efficiency
    
    # Observation error covariance
    R = observation_variance * np.eye(n_obs)
    
    # Background error covariance at observed locations
    # B_oo = ens_centered[:, observed_indices].T @ ens_centered[:, observed_indices] / (n_members - 1)
    X_o = ens_centered[:, observed_indices]  # (n_members, n_obs)
    B_oo = (X_o.T @ X_o) / (n_members - 1)
    
    # Add localization to B_oo
    B_oo = _localize_cov(B_oo, observed_indices, alpha=3.0)
    
    # Compute Kalman gain for observed variables
    # K = B_oo @ inv(B_oo + R)
    # For numerical stability, use Cholesky or SVD
    try:
        # B_oo + R should be positive definite
        S = B_oo + R
        # Use SVD for stability
        U, s, Vt = np.linalg.svd(S)
        # Pseudo-inverse
        S_inv = Vt.T @ np.diag(1.0 / s) @ U.T
        K = B_oo @ S_inv  # (n_obs, n_obs)
    except np.linalg.LinAlgError:
        # Fallback: use diagonal approximation
        K = np.diag(np.diag(B_oo) / (np.diag(B_oo) + observation_variance))
    
    # Update observed variables in ensemble
    # For each member, update observed coordinates
    obs_mean = np.mean(observations, axis=0)  # (n_obs,)
    
    # Innovation
    innovation = observations - (ens_centered[:, observed_indices] + ens_mean[observed_indices])
    # innovation: (n_members, n_obs)
    
    # Update: x_o_new = x_o + K @ innovation
    # But we need to be careful: the standard EnSRF update
    # Let's do it member-by-member for the observed part
    
    updated_ensemble = ensemble.copy()
    
    # First, update the observed variables
    for m in range(n_members):
        innov_m = observations[m] - (ensemble[m, observed_indices] - ens_mean[observed_indices])
        # Actually, innovation should be: obs - forecast
        # forecast_m = ensemble[m, observed_indices]
        # innov_m = observations[m] - forecast_m
        innov_m = observations[m] - ensemble[m, observed_indices]
        updated_ensemble[m, observed_indices] += K @ innov_m
    
    # Now update unobserved variables using regression
    # For each unobserved variable j, compute regression coefficients
    # on observed variables using the ensemble
    for j in unobserved_indices:
        if time.time() > deadline - 5.0:
            break
        
        # Cross-covariance between variable j and observed variables
        # B_jo = ens_centered[:, j] @ X_o.T / (n_members - 1)
        x_j = ens_centered[:, j]  # (n_members,)
        B_jo = (x_j @ X_o.T) / (n_members - 1)  # (1, n_obs)
        
        # Localize cross-covariance
        # Apply distance-based localization
        idx_j = j
        dist_j = np.abs(idx_j - observed_indices)
        dist_j = np.minimum(dist_j, n_vars - dist_j)
        loc_mask = np.exp(-dist_j / 3.0)
        B_jo = B_jo * loc_mask
        
        # Regression coefficient: a = B_jo @ B_oo_inv
        # a_j = B_jo @ inv(B_oo + R) ... actually for regression we use B_oo_inv
        # In the analysis, the update for unobserved is:
        # x_j_new = x_j + a_j @ (x_o_new - x_o)
        # where a_j = B_jo @ B_oo^{-1}
        
        try:
            # a = B_jo @ inv(B_oo)
            B_oo_inv = np.linalg.inv(B_oo)
            a = B_jo @ B_oo_inv  # (1, n_obs)
        except np.linalg.LinAlgError:
            # Fallback: use only nearest neighbors
            a = np.zeros(n_obs)
            # Find nearest observed index
            nearest = observed_indices[np.argmin(dist_j)]
            nearest_idx = np.where(observed_indices == nearest)[0][0]
            # Simple regression coefficient
            var_j = np.var(ens_centered[:, j])
            var_o = np.var(X_o[:, nearest_idx])
            if var_o > 1e-10:
                cov_jo = np.cov(ens_centered[:, j], X_o[:, nearest_idx])[0, 1]
                a[nearest_idx] = cov_jo / var_o
        
        # Compute the change in observed variables for each member
        # delta_x_o = updated_ensemble[:, observed_indices] - ensemble[:, observed_indices]
        delta_x_o = updated_ensemble[:, observed_indices] - ensemble[:, observed_indices]
        
        # Update unobserved variable
        # x_j_new = x_j + a @ delta_x_o
        update = delta_x_o @ a.T  # (n_members,)
        updated_ensemble[:, j] += update
    
    # Apply inflation to prevent ensemble collapse
    # Simple multiplicative inflation based on spread
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_centered = updated_ensemble - updated_mean
    
    # Compute spread
    spread = np.std(updated_centered, axis=0)
    target_spread = 2.0  # heuristic
    
    # Inflate if spread is too small
    for j in range(n_vars):
        if spread[j] < target_spread:
            inflation = target_spread / (spread[j] + 1e-10)
            inflation = min(inflation, 2.0)  # cap inflation
            updated_ensemble[:, j] = updated_mean[j] + inflation * updated_centered[:, j]
    
    # Clamp to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finite
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble

# EVOLVE-BLOCK-END
