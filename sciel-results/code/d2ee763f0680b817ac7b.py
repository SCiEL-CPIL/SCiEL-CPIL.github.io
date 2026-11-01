"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad_flat(x):
    """
    Compute logarithmic energy and gradient for flattened array of N points.
    x is shape (3*N,).
    """
    pts = x.reshape(N, 3)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0)
    dist_safe = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(N, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad_pts = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad_pts.flatten()

def _sphere_constraints(x):
    """
    Constraint: each point must lie on the unit sphere.
    g_i(x) = ||x_i||^2 - 1 = 0
    """
    pts = x.reshape(N, 3)
    norms_sq = np.sum(pts * pts, axis=1)
    return norms_sq - 1.0

def _sphere_jacobian(x):
    """
    Jacobian of the constraints.
    J[i, 3j:3j+3] = 2 * x_j for constraint i
    """
    pts = x.reshape(N, 3)
    J = np.zeros((N, 3 * N))
    for i in range(N):
        J[i, 3*i:3*i+3] = 2.0 * pts[i]
    return J

def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    # Initial configuration: Fibonacci spiral
    pts_init = _fibonacci_spiral(N)
    x0 = pts_init.flatten()
    
    # Define constraints
    constraint = NonlinearConstraint(_sphere_constraints, 0.0, 0.0, jac=_sphere_jacobian)
    
    # Optimize using Trust-Region Constrained method
    try:
        res = minimize(
            _log_energy_and_grad_flat, 
            x0, 
            method='trust-constr', 
            constraints=[constraint],
            options={'maxiter': 1000, 'gtol': 1e-8, 'verbose': 0}
        )
        
        if res.success or res.fun < _log_energy_and_grad_flat(x0)[0]:
            best_x = res.x
        else:
            best_x = x0
    except Exception:
        best_x = x0
        
    pts_best = best_x.reshape(N, 3)
    
    # Project to sphere to ensure numerical stability
    norms = np.linalg.norm(pts_best, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts_best = pts_best / norms
    
    # Check for close points
    dist = np.linalg.norm(pts_best[:, np.newaxis, :] - pts_best[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        pts_best = _fibonacci_spiral(N)
        
    if not np.all(np.isfinite(pts_best)):
        pts_best = _fibonacci_spiral(N)
        
    return pts_best
# EVOLVE-BLOCK-END
