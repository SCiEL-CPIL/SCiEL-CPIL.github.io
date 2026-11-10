"""Rewritten program for logarithmic energy minimization."""
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

N = 150
DEADLINE = 120.0
SAFETY_MARGIN = 0.95

def _log_energy_precise(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _fibonacci_spiral(n, offset=0.0):
    """Golden-angle spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _cartesian_energy_grad(pts):
    """
    Calculate Energy and Gradient in Cartesian coordinates.
    pts: (N, 3) array of points on the sphere.
    Returns: (energy, gradient) where gradient is (N, 3).
    """
    n = pts.shape[0]
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    # Inverse squared distances
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    
    # Gradient of E w.r.t x_i is - sum_j (x_i - x_j) / r_ij^2
    # Note: The force vector from j to i is (x_i - x_j).
    # The potential is log(1/r) = -log(r).
    # d(-log(r))/dr = -1/r.
    # dr/dx_i = (x_i - x_j)/r.
    # dE/dx_i = -1/r * (x_i - x_j)/r = -(x_i - x_j)/r^2.
    # So grad_i = - sum_{j != i} (x_i - x_j) / r_ij^2
    # Which is sum_{j != i} (x_j - x_i) / r_ij^2
    # Using diff = x_i - x_j, grad = - sum diff * inv_r2
    
    grad = -np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    # Energy calculation
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    log_r = np.log(r)
    np.fill_diagonal(log_r, 0.0)
    energy = -0.5 * np.sum(log_r) # Sum over i<j is half the sum over i!=j
    
    return energy, grad

def _optimize_lbfgs_cartesian(pts, deadline):
    """
    Optimize points in Cartesian space using L-BFGS-B.
    We treat the 3N coordinates as free variables but project back to sphere
    inside the function call. However, L-BFGS-B expects a smooth manifold.
    A better approach for L-BFGS on sphere is to use the tangent space or
    simply rely on the fact that if we start on the sphere and take small steps,
    we stay close. But standard L-BFGS in R^3 will drift off the sphere.
    
    Alternative: Use the spherical parameterization for L-BFGS as in the seed,
    but it had issues. Let's try a hybrid:
    1. Coarse optimization with L-BFGS in Cartesian, but we must ensure we stay on sphere.
       Actually, L-BFGS in Cartesian without constraint projection is bad.
       
       Let's stick to the Spherical Coordinate L-BFGS but improve it, OR
       use a custom Riemannian Gradient Descent for the whole process if it's fast enough.
       
       Given N=150, N^2=22500 operations per step.
       1000 steps = 22.5M ops. Very fast in NumPy.
       
       Let's use a robust Gradient Descent with adaptive step size for the main optimization,
       as it naturally handles the constraint if we project after each step.
    """
    pass # Placeholder, we will implement a custom GD loop in construct

def _gradient_descent_polish(pts, deadline, initial_lr=1e-3, min_lr=1e-8):
    """
    High-precision gradient descent polish.
    """
    pts = pts.copy().astype(np.float64)
    current_energy = _log_energy_precise(pts) # Use precise energy for tracking, but grad from vectorized calc
    # Actually, using precise energy for every step is too slow (O(N^2) python loop).
    # We use the vectorized energy for the gradient, and check precise energy periodically.
    
    e_vec, grad = _cartesian_energy_grad(pts)
    current_energy_vec = e_vec
    
    lr = initial_lr
    no_improve_count = 0
    
    while time.time() < deadline:
        # Calculate gradient
        e_vec, grad = _cartesian_energy_grad(pts)
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        # Step
        pts_new = pts - lr * grad
        pts_new = _project_to_sphere(pts_new)
        
        # Check validity
        # If points get too close, energy explodes. 
        # We can check min distance.
        diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-9:
            # Step was too large or collapsed, reduce LR and retry
            lr *= 0.5
            continue
            
        e_new_vec, _ = _cartesian_energy_grad(pts_new)
        
        if e_new_vec < current_energy_vec:
            pts = pts_new
            current_energy_vec = e_new_vec
            no_improve_count = 0
            # Decay LR slowly
            lr *= 0.99
        else:
            no_improve_count += 1
            lr *= 0.5
            if no_improve_count > 20:
                # Stuck or converged, reduce LR further and reset counter
                lr *= 0.5
                no_improve_count = 0
                if lr < min_lr:
                    break
                    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = 1e30
    
    # Generate initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    inits.append(_fibonacci_spiral(N, offset=0.5))
    inits.append(_fibonacci_spiral(N, offset=1.0))
    inits.append(_random_points(N, rng))
    inits.append(_random_points(N, rng))
    
    # Phase 1: Coarse Optimization
    # We will use L-BFGS in Spherical Coordinates for speed, as it is robust enough for coarse search.
    # Then we switch to Cartesian GD for polish.
    
    def _spherical_to_cartesian(theta, phi):
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        return np.column_stack([sin_t * cos_p, sin_t * sin_p, cos_t])

    def _cartesian_to_spherical(pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        return np.column_stack([theta, phi])

    def _spherical_energy_grad(params):
        theta = params[0::2]
        phi = params[1::2]
        X = _spherical_to_cartesian(theta, phi)
        # Use Cartesian energy/grad and chain rule
        e_cart, grad_cart = _cartesian_energy_grad(X)
        
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        dX_dtheta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
        dX_dphi = np.column_stack([-sin_t * sin_p, sin_t * cos_p, np.zeros(N)])
        
        grad_theta = np.sum(grad_cart * dX_dtheta, axis=1)
        grad_phi = np.sum(grad_cart * dX_dphi, axis=1)
        
        grad = np.empty(2 * N)
        grad[0::2] = grad_theta
        grad[1::2] = grad_phi
        
        return e_cart, grad

    for init in inits:
        if time.time() > deadline - 10: # Leave 10s for final polish
            break
            
        # Convert to spherical
        theta_phi = _cartesian_to_spherical(init)
        x0 = theta_phi.ravel()
        
        try:
            res = minimize(
                _spherical_energy_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxfun': 500
                }
            )
            x_opt = res.x
        except Exception:
            continue
            
        theta_opt = x_opt[0::2]
        phi_opt = x_opt[1::2]
        X_opt = _spherical_to_cartesian(theta_opt, phi_opt)
        X_opt = _project_to_sphere(X_opt)
        
        # Check validity
        diff = X_opt[:, np.newaxis, :] - X_opt[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-9:
            continue
            
        e = _log_energy_precise(X_opt)
        
        if e < best_energy:
            best_energy = e
            best_pts = X_opt.copy()
            
    if best_pts is None:
        # Fallback
        best_pts = _project_to_sphere(_fibonacci_spiral(N))
        
    # Phase 2: High-Precision Polish
    # Use remaining time for gradient descent polish on the best solution
    if time.time() < deadline:
        polished_pts = _gradient_descent_polish(best_pts, deadline, initial_lr=1e-4, min_lr=1e-10)
        
        # Check validity of polished points
        diff = polished_pts[:, np.newaxis, :] - polished_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 1e-9:
            e_polished = _log_energy_precise(polished_pts)
            if e_polished < best_energy:
                best_pts = polished_pts
                best_energy = e_polished

    return best_pts
