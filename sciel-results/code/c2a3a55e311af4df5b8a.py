"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    and its gradient w.r.t. pts.
    
    d_ij = |p_i - p_j| (chordal distance on R^3, points on unit sphere)
    grad_i E = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    """
    pts = np.asarray(pts)
    n = len(pts)
    # Pairwise differences
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    # Squared distances
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Distances
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # We only want upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i E = sum_{j != i} (p_i - p_j) / |p_i - p_j|^2
    # diff[i, j] = p_i - p_j
    # term[i, j] = diff[i, j] / dist_sq[i, j]
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = diff / dist_sq[:, :, np.newaxis]
    # Set diagonal to zero
    idx = np.diag_indices(n)
    terms[idx[0], idx[1], :] = 0.0
    # Sum over j for each i
    grad = np.sum(terms, axis=1)  # (n, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_optimize(pts0, maxiter=5000, tol=1e-10):
    """
    Optimize points on the sphere using projected gradient / L-BFGS-B.
    We use a parameterization-free approach: optimize in R^{3n} with a
    penalty for deviation from the sphere, or use a retraction-based method.
    
    Here we use L-BFGS-B on the ambient space with a strong sphere constraint
    via projection at each step (via a wrapper).
    
    Actually, a cleaner approach: use scipy minimize with 'L-BFGS-B' on the
    unconstrained ambient coordinates, but project to sphere after each
    gradient evaluation. This is a heuristic but often works.
    
    Better: use a parameterization. For 100 points, we can use spherical
    coordinates, but that has singularities. 
    
    Alternative: use the fact that the manifold is the product of spheres.
    We can use a simple projected gradient descent with line search.
    
    Let's use a simple approach: run L-BFGS-B on ambient space, and at each
    iteration, project the points to the sphere. The gradient we provide is
    the ambient gradient (not projected), which is a heuristic.
    """
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, _ = _energy_and_grad(pts)
        return e
    
    def gradient(x):
        pts = x.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space of sphere at each point
        # Tangent projection: g_tan = g - (g . p) * p
        g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tan.flatten()
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': tol * 1e-2, 'maxls': 50}
    )
    
    pts_opt = _project_to_sphere(result.x.reshape(-1, 3))
    return pts_opt, result.fun


def _anneal_and_optimize(pts0, n_restarts=5, time_budget=100):
    """
    Try multiple restarts with different initial conditions and local optimization.
    """
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        if restart == 0:
            pts_init = pts0
        else:
            # Perturb the best solution found so far
            pts_init = best_pts + 0.1 * np.random.randn(*best_pts.shape)
            pts_init = _project_to_sphere(pts_init)
        
        pts_opt, energy = _local_optimize(pts_init, maxiter=3000, tol=1e-10)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    return best_pts, best_energy


def construct():
    """
    Construct 100 points on the unit sphere minimizing logarithmic energy.
    """
    start_time = time.time()
    deadline = 110  # Leave 10s margin before 120s limit
    
    # Initial points: Fibonacci sphere
    pts0 = _fibonacci_sphere(N)
    
    # Run optimization
    pts_opt, energy = _anneal_and_optimize(pts0, n_restarts=8, time_budget=deadline - (time.time() - start_time))
    
    # Ensure valid output
    pts_opt = _project_to_sphere(pts_opt)
    
    # Check for non-finite values
    if not np.all(np.isfinite(pts_opt)):
        pts_opt = _fibonacci_sphere(N)
    
    # Check minimum distance
    diff = pts_opt[:, np.newaxis, :] - pts_opt[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        pts_opt = _fibonacci_sphere(N)
    
    return pts_opt
# EVOLVE-BLOCK-END
