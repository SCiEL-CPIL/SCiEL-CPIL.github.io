"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 120.0
SAFETY_MARGIN = 0.90  # use 90% of time

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
    """Golden-angle spiral points on unit sphere with optional phase offset."""
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

def _repulsion_refine_cartesian(pts, n_iters=200, lr=0.01, deadline=None):
    """
    Iterative repulsion in Cartesian space to spread points.
    This acts as a pre-conditioner before spherical optimization.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        forces = np.zeros_like(pts)
        # Vectorized force calculation for speed
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0) # Avoid div by zero
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        # F_i = sum_j (x_i - x_j) / d_ij^2
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        pts += lr * forces
        # Project back to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts

def _cartesian_to_spherical(pts):
    """Convert Cartesian points to (theta, phi) array of shape (N, 2)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.column_stack([theta, phi])

def _spherical_energy_grad(params):
    """
    Calculate Energy and Gradient for spherical coordinates.
    params: array of shape (2*N), alternating [theta_0, phi_0, theta_1, phi_1, ...]
    Returns: (energy, gradient)
    """
    theta = params[0::2]
    phi = params[1::2]
    
    # Convert to Cartesian
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    X = np.column_stack([sin_t * cos_p, sin_t * sin_p, cos_t])
    
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    
    # Squared distances
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Inverse squared distances
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    
    # Gradient of E w.r.t Cartesian X
    # dE/dX_i = - sum_j (X_i - X_j) / r_ij^2
    grad_cart = -np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    # Energy
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    log_r = np.log(r)
    np.fill_diagonal(log_r, 0.0)
    energy = -0.5 * np.sum(log_r)
    
    # Jacobian dX/dparams
    dX_dtheta = np.column_stack([
        cos_t * cos_p, 
        cos_t * sin_p, 
        -sin_t
    ])
    
    dX_dphi = np.column_stack([
        -sin_t * sin_p, 
        sin_t * cos_p, 
        np.zeros(N)
    ])
    
    # Chain rule
    grad_theta = np.sum(grad_cart * dX_dtheta, axis=1)
    grad_phi = np.sum(grad_cart * dX_dphi, axis=1)
    
    # Pack gradient
    grad = np.empty(2 * N)
    grad[0::2] = grad_theta
    grad[1::2] = grad_phi
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = []
    # Fibonacci spiral variants
    inits.append(_fibonacci_spiral(N))
    inits.append(_fibonacci_spiral(N, offset=0.5))
    inits.append(_fibonacci_spiral(N, offset=1.0))
    # Random starts
    for _ in range(3):
        inits.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = 1e30
    
    # For each initialization:
    # 1. Pre-condition with Cartesian repulsion
    # 2. Convert to spherical
    # 3. Optimize with L-BFGS-B
    for init in inits:
        if time.time() > deadline:
            break
            
        # 1. Pre-condition
        refined_cart = _repulsion_refine_cartesian(init, n_iters=100, lr=0.01, deadline=deadline)
        
        # 2. Convert to spherical
        theta_phi = _cartesian_to_spherical(refined_cart)
        x0 = theta_phi.ravel()
        
        # 3. Optimize
        try:
            res = minimize(
                _spherical_energy_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': 1000
                }
            )
            
            # Convert result back to Cartesian
            theta_opt = res.x[0::2]
            phi_opt = res.x[1::2]
            
            sin_t = np.sin(theta_opt)
            cos_t = np.cos(theta_opt)
            cos_p = np.cos(phi_opt)
            sin_p = np.sin(phi_opt)
            
            X_opt = np.column_stack([sin_t * cos_p, sin_t * sin_p, cos_t])
            
            # Normalize
            norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_opt = X_opt / norms
            
            # Check for validity (no points too close)
            min_dist = np.inf
            # Quick check using vectorized distance
            diff = X_opt[:, np.newaxis, :] - X_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 1e-9:
                continue
                
            # Calculate precise energy
            e = _log_energy_precise(X_opt)
            
            if e < best_energy:
                best_energy = e
                best_pts = X_opt.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        # Fallback to best pre-conditioned point or initial
        best_pts = _repulsion_refine_cartesian(_fibonacci_spiral(N), n_iters=500, lr=0.01, deadline=deadline)
        
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
