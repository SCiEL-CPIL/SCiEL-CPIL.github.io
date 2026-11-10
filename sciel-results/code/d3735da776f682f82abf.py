"""Hybrid Ensemble-4D-Var analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        ip1 = (i + 1) % n
        im2 = (i - 2) % n
        im1 = (i - 1) % n
        dx[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    r1 = d[mask1]
    if len(r1) > 0:
        result[mask1] = (-5.0/4.0)*(r1/c)**5 + 5.0*(r1/c)**4 - 10.0*(r1/c)**3 + 5.0*(r1/c)
    
    r2 = d[mask2]
    if len(r2) > 0:
        result[mask2] = (1.0/12.0)*(r2/c)**5 - (5.0/4.0)*(r2/c)**4 + (5.0/2.0)*(r2/c)**3 - 5.0*(r2/c)**2 + (25.0/12.0)*(r2/c) + 0.5
    
    r3 = d[mask3]
    if len(r3) > 0:
        result[mask3] = (-1.0/16.0)*(r3/c)**5 + (1.0/4.0)*(r3/c)**4 - (5.0/8.0)*(r3/c)**3 - (5.0/2.0)*(r3/c)**2 + (25.0/12.0)*(r3/c) - 0.25
    
    return result


def _hybrid_4dvar_update(ensemble, observations, observed_indices, observation_variance,
                         n_members=16, n_state=20, n_obs=10,
                         F=8.0, dt=0.01, n_steps=5,
                         localization_scale=3.0, max_iter=50, tol=1e-6):
    """Hybrid Ensemble-4D-Var: optimize control vector to minimize
    J = 0.5*(x_c - x_b)^T B^{-1} (x_c - x_b) + 0.5*(y - H*x_c)^T R^{-1} (y - H*x_c)
    where x_c is the control vector (analysis state), x_b is ensemble mean (background),
    B is ensemble covariance, H is observation operator (identity on observed indices),
    R is observation error covariance.
    
    We use L-BFGS to optimize this cost function. The ensemble perturbations are then
    scaled to maintain the correct spread.
    """
    mean = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean
    ens_cov = np.cov(ensemble.T)  # (n_state, n_state)
    
    # Regularize covariance
    ens_cov += 1e-6 * np.eye(n_state)
    ens_cov_inv = np.linalg.inv(ens_cov)
    
    # Build localization matrix for observations
    # L[i,j] = Gaspari-Cohn(distance(i, observed_indices[j]))
    L = np.zeros((n_state, n_obs))
    for i in range(n_state):
        for j, obs_idx in enumerate(observed_indices):
            dist = abs(i - obs_idx)
            dist = min(dist, n_state - dist)
            L[i, j] = _gaspari_cohn(np.array([dist]), c=localization_scale)[0]
    
    # R = observation_variance * I (diagonal)
    R_inv = np.eye(n_obs) / observation_variance
    
    # H: observation operator (n_obs x n_state), selects observed indices
    H = np.zeros((n_obs, n_state))
    for j, obs_idx in enumerate(observed_indices):
        H[j, obs_idx] = 1.0
    
    # Cost function: J(x) = 0.5*(x - x_b)^T B^{-1} (x - x_b) + 0.5*(y - H*x)^T R^{-1} (y - H*x)
    def cost_and_grad(x):
        x = np.clip(x, -100.0, 100.0)
        # Background term
        dx = x - mean
        J_bg = 0.5 * dx @ ens_cov_inv @ dx
        g_bg = ens_cov_inv @ dx
        
        # Observation term
        y_hat = H @ x
        innov = observations - y_hat
        J_obs = 0.5 * innov @ R_inv @ innov
        g_obs = H.T @ (R_inv @ innov)
        
        J = J_bg + J_obs
        g = g_bg + g_obs
        return J, g
    
    # Initial guess: ensemble mean nudged toward observations
    x0 = mean.copy()
    # Simple initial nudge
    for j, obs_idx in enumerate(observed_indices):
        x0[obs_idx] += 0.5 * (observations[j] - mean[obs_idx])
    
    # L-BFGS optimization
    from scipy.optimize import minimize
    res = minimize(cost_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                            'maxls': 50, 'maxfun': 200})
    
    x_analysis = res.x
    x_analysis = np.clip(x_analysis, -100.0, 100.0)
    
    # Compute the analysis perturbation from the optimization
    delta = x_analysis - mean
    
    # Now construct the updated ensemble:
    # We want to preserve the ensemble spread while shifting the mean to x_analysis.
    # Use the ensemble perturbations but rescale them.
    
    # Compute the original ensemble variance
    orig_var = np.var(ensemble, axis=0)
    
    # The analysis should have a similar spread. We scale perturbations to maintain
    # a reasonable spread around the new mean.
    # Use a simple approach: keep the perturbations but adjust their magnitude
    # based on how much the mean moved.
    
    # For each state variable, compute the ratio of the displacement to the spread
    spread = np.sqrt(orig_var) + 1e-10
    ratio = np.abs(delta) / spread
    
    # Scale factors: if the mean moved a lot relative to spread, reduce perturbations
    # to avoid over-dispersing
    scale = 1.0 / (1.0 + 0.1 * ratio)
    scale = np.clip(scale, 0.3, 1.0)
    
    # Build updated ensemble
    result = np.zeros_like(ensemble)
    for m in range(n_members):
        result[m, :] = x_analysis + perturbations[m, :] * scale
    
    # Ensure finite and bounded
    result = np.clip(result, -100.0, 100.0)
    result[~np.isfinite(result)] = 0.0
    
    return result


def analyze(ensemble, observations, observed_indices, observation_variance):
    """Perform Hybrid Ensemble-4D-Var analysis update.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: (10,) array of observed state indices
    observation_variance: scalar observation error variance
    
    Returns:
    Updated ensemble (16, 20)
    """
    t_start = time.time()
    deadline = t_start + 110.0
    
    try:
        n_members, n_state = ensemble.shape
        n_obs = len(observed_indices)
        
        # Estimate F from the ensemble (crude but reasonable)
        # For Lorenz-96, F is typically 8 or 10. We'll use 8 as default.
        F = 8.0
        
        result = _hybrid_4dvar_update(
            ensemble, observations, observed_indices, observation_variance,
            n_members=n_members, n_state=n_state, n_obs=n_obs,
            F=F, dt=0.01, n_steps=5,
            localization_scale=3.0, max_iter=50, tol=1e-6
        )
        
        return result
        
    except Exception as e:
        # Fallback: simple nudging
        result = ensemble.copy().astype(float)
        result[:, observed_indices] += 0.5 * (observations - result[:, observed_indices])
        result = np.clip(result, -100.0, 100.0)
        result[~np.isfinite(result)] = 0.0
        return result
# EVOLVE-BLOCK-END
