"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import basinhopping, minimize

def _cartesian_from_spherical(theta, phi):
    """Convert spherical coordinates to Cartesian coordinates on the unit sphere."""
    sin_theta = np.sin(theta)
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack((x, y, z))

def _energy_and_grad_spherical(params):
    """
    Compute logarithmic energy and its gradient with respect to spherical parameters.
    params: array of shape (2*N,), alternating theta, phi.
    """
    n = len(params) // 2
    theta = params[0::2]
    phi = params[1::2]
    
    # Compute Cartesian coordinates
    pts = _cartesian_from_spherical(theta, phi)
    
    # Compute pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist_sq_safe = np.where(mask, dist_sq, 1.0)
    dist = np.sqrt(dist_sq_safe)
    
    # Energy: sum_{i<j} log(1/dist)
    # We need to sum only i < j to avoid double counting
    # log(1/dist) = -log(dist)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    
    # Sum over upper triangle
    triu_indices = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[triu_indices].tolist())
    
    # Gradient calculation
    # g_cart[i] = -sum_{j != i} (pts[i] - pts[j]) / |pts[i] - pts[j]|^2
    # Note: The derivative of log(1/|x-y|) w.r.t x is - (x-y)/|x-y|^2
    # So grad_E_x = - sum_j (x_i - x_j) / |x_i - x_j|^2
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq_safe[mask]
    
    # weighted_diff[i, j] = (pts[i] - pts[j]) * inv_dist_sq[i, j]
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j for each i
    # This gives -dE/dx_i
    grad_neg_cart = np.sum(weighted_diff, axis=1)
    
    # True Cartesian gradient
    grad_cart = -grad_neg_cart
    
    # Project to spherical coordinates
    # dE/dtheta_i = grad_cart[i] . d(pts_i)/dtheta_i
    # dE/dphi_i = grad_cart[i] . d(pts_i)/dphi_i
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    # d(pts)/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    dpts_dtheta = np.column_stack((cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta))
    
    # d(pts)/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    dpts_dphi = np.column_stack((-sin_theta * sin_phi, sin_theta * cos_phi, np.zeros(n)))
    
    grad_theta = np.sum(grad_cart * dpts_dtheta, axis=1)
    grad_phi = np.sum(grad_cart * dpts_dphi, axis=1)
    
    # Interleave for the params vector
    grad_params = np.empty(2 * n)
    grad_params[0::2] = grad_theta
    grad_params[1::2] = grad_phi
    
    return energy, grad_params

def _objective_and_grad(params):
    """Wrapper for scipy minimize, returns (f, g)."""
    return _energy_and_grad_spherical(params)

def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_limit = 115.0
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    
    # Convert to spherical coords
    theta_init = np.arccos(z)
    phi_init = phi
    
    # Interleave
    params_init = np.empty(2 * N)
    params_init[0::2] = theta_init
    params_init[1::2] = phi_init
    
    best_result = None
    best_energy = float('inf')
    
    # Run Basin Hopping
    # We define a step size for the random perturbations
    # Typical step size for angles is around 0.1 to 1.0 radians
    step_size = 0.5 
    
    try:
        # Use L-BFGS-B for local minimization
        # We provide the gradient, so it will be efficient
        result = basinhopping(
            _objective_and_grad, 
            params_init, 
            niter=100, 
            stepsize=step_size, 
            miniter=100, 
            disp=False, 
            maxfun=100000,
            method='L-BFGS-B'
        )
        
        if result.success or result.fun < best_energy:
            best_energy = result.fun
            best_result = result.x
            
    except Exception as e:
        # Fallback if basin hopping fails or times out
        pass

    # If we have time, do a final polish with L-BFGS-B directly
    if time.time() - start_time < time_limit - 10:
        current_params = best_result if best_result is not None else params_init
        try:
            res = minimize(
                _objective_and_grad, 
                current_params, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_result = res.x
        except Exception:
            pass

    # Convert best spherical params back to Cartesian
    if best_result is not None:
        theta = best_result[0::2]
        phi = best_result[1::2]
        pts = _cartesian_from_spherical(theta, phi)
    else:
        # Fallback to initial guess
        pts = _cartesian_from_spherical(theta_init, phi_init)
        
    # Ensure points are on sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts

# EVOLVE-BLOCK-END
