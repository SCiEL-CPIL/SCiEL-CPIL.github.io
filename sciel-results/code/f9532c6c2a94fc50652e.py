# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log, sqrt
import time
import scipy.optimize

N = 100

def _cartesian_from_angles(params):
    """Convert [theta_1, phi_1, ..., theta_N, phi_N] to (N, 3) Cartesian."""
    thetas = params[0::2]
    phis = params[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])

def _log_energy_and_grad_cartesian(points):
    """Compute log energy and gradient in Cartesian space."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 1e-30, 4.0, out=d2)
    d = np.sqrt(d2)
    
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    energy = fsum(-np.log(dists).tolist())
    
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    grad_cart = row_sums * points - (M @ points)
    
    return energy, grad_cart

def _energy_and_grad_params(params):
    """Compute energy and gradient w.r.t. parameters [theta, phi]."""
    points = _cartesian_from_angles(params)
    energy, grad_cart = _log_energy_and_grad_cartesian(points)
    
    thetas = params[0::2]
    phis = params[1::2]
    
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    sin_p = np.sin(phis)
    cos_p = np.cos(phis)
    
    grad_theta = grad_cart[:, 0] * (cos_t * cos_p) + grad_cart[:, 1] * (cos_t * sin_p) + grad_cart[:, 2] * (-sin_t)
    grad_phi = grad_cart[:, 0] * (-sin_t * sin_p) + grad_cart[:, 1] * (sin_t * cos_p)
    
    grad_params = np.empty_like(params)
    grad_params[0::2] = grad_theta
    grad_params[1::2] = grad_phi
    
    return energy, grad_params

def _optimize_single(params0, maxiter=300):
    """Run L-BFGS-B optimization."""
    try:
        result = scipy.optimize.minimize(
            _energy_and_grad_params, 
            params0, 
            method='L-BFGS-B', 
            jac=True, 
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x, result.fun
    except Exception:
        return params0, _energy_and_grad_params(params0)[0]

def construct():
    rng = np.random.default_rng(42)
    best_energy = float('inf')
    best_params = None
    
    # Multiple restarts with different initializations
    for i in range(10):
        # Mix of random and structured initializations
        if i == 0:
            # Fibonacci-like spiral in angles
            thetas = np.linspace(0, np.pi, N, endpoint=False) + np.pi / N
            phis = np.linspace(0, 2 * np.pi, N, endpoint=False)
        else:
            thetas = rng.uniform(0, np.pi, N)
            phis = rng.uniform(0, 2 * np.pi, N)
        
        params0 = np.zeros(2 * N)
        params0[0::2] = thetas
        params0[1::2] = phis
        
        params, energy = _optimize_single(params0, maxiter=200)
        if energy < best_energy:
            best_energy = energy
            best_params = params

    if best_params is None:
        thetas = np.linspace(0, np.pi, N, endpoint=False) + 0.1
        phis = np.linspace(0, 2 * np.pi, N)
        best_params = np.zeros(2 * N)
        best_params[0::2] = thetas
        best_params[1::2] = phis

    points = _cartesian_from_angles(best_params)
    return points
# EVOLVE-BLOCK-END
