"""Rewritten program for minimizing logarithmic energy of 150 points on the sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

DEADLINE = 145.0  # 145s out of 150s wall-clock limit
SAFETY_MARGIN = 0.92  # use 92% of the deadline

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

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient w.r.t. Cartesian coordinates.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)
    np.fill_diagonal(d, 1.0)
    
    # Energy
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _projected_energy_and_grad(params):
    """
    Energy and gradient for optimization in Cartesian space with projection.
    params: flat array of shape (3*N)
    """
    pts = params.reshape(N, 3)
    energy, grad = _energy_and_grad(pts)
    # Project gradient to tangent space: grad_tangent = grad - (grad·x)x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    return energy, grad_tangent.ravel()

def _repulsion_refine_cartesian(pts, n_iters=200, lr=0.005, deadline=None):
    """
    Iterative repulsion in Cartesian space to spread points.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        pts += lr * forces
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
    return pts

def _optimize_cartesian(x0, deadline, maxiter=1000):
    """Run L-BFGS-B optimization in Cartesian coordinates with projection."""
    try:
        res = minimize(
            _projected_energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 2000
            }
        )
        return res.x
    except Exception:
        return x0

def _polish(x0, deadline, maxiter=300):
    """High-precision polish step with tighter tolerances."""
    try:
        res = minimize(
            _projected_energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxfun': 1000
            }
        )
        return res.x
    except Exception:
        return x0

def _check_validity(pts):
    """Check if points are valid (no two points closer than 1e-9)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min_dist >= 1e-9

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
    inits.append(_fibonacci_spiral(N, offset=2.0))
    # Random starts
    for _ in range(4):
        inits.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = 1e30
    best_params = None
    
    # For each initialization:
    # 1. Pre-condition with Cartesian repulsion
    # 2. Optimize with L-BFGS-B in Cartesian space
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # 1. Pre-condition
        refined_cart = _repulsion_refine_cartesian(init, n_iters=100, lr=0.01, deadline=deadline)
        
        # 2. Optimize
        x0 = refined_cart.ravel()
        x_opt = _optimize_cartesian(x0, deadline)
        
        # Convert result back to Cartesian and normalize
        X_opt = x_opt.reshape(N, 3)
        norms = np.linalg.norm(X_opt, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_opt = X_opt / norms
        
        # Check for validity
        if not _check_validity(X_opt):
            continue
        
        # Calculate precise energy
        e = _log_energy_precise(X_opt)
        
        if e < best_energy:
            best_energy = e
            best_pts = X_opt.copy()
            best_params = x_opt.copy()
    
    # Final Polish Phase
    if best_params is not None and time.time() < deadline - 5:
        x_polished = _polish(best_params, deadline)
        
        X_polished = x_polished.reshape(N, 3)
        norms = np.linalg.norm(X_polished, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_polished = X_polished / norms
        
        if _check_validity(X_polished):
            e_polished = _log_energy_precise(X_polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = X_polished.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
