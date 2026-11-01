"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    Constraint: each point is on the unit sphere (handled by projection in the optimizer).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances
    # dist[i,j] = ||pts[i] - pts[j]||
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # We only want upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -np.sum(np.log(d_upper))
    
    # Gradient: dE/dpts[i] = sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2 * (-1) * (-1)
    # Wait: E = -sum_{i<j} log(d_ij)
    # dE/dpts[i] = -sum_{j != i, j>i} d/dpts[i] log(||pts[i]-pts[j]||) - sum_{j != i, j<i} d/dpts[i] log(||pts[i]-pts[j]||)
    # d/dpts[i] log(||pts[i]-pts[j]||) = (pts[i]-pts[j]) / ||pts[i]-pts[j]||^2
    # So dE/dpts[i] = -sum_{j != i} (pts[i]-pts[j]) / ||pts[i]-pts[j]||^2
    
    # Compute gradient efficiently
    # grad[i] = -sum_{j != i} (pts[i] - pts[j]) / dist[i,j]^2
    # We can compute this as: for each i, sum over j != i of (pts[i] - pts[j]) / dist[i,j]^2
    
    # dist_safe^2
    dist_sq = dist_safe ** 2
    
    # Create a matrix W where W[i,j] = 1/dist[i,j]^2 if i != j, else 0
    W = np.zeros_like(dist_sq)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad[i] = -sum_j W[i,j] * (pts[i] - pts[j])
    # = -sum_j W[i,j]*pts[i] + sum_j W[i,j]*pts[j]
    # = -pts[i] * sum_j W[i,j] + (W @ pts)[i]
    
    row_sum = W.sum(axis=1)  # shape (n,)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _objective_and_grad(x_flat):
    """
    Objective function for scipy minimize.
    x_flat: flattened N x 3 array.
    Returns (energy, gradient) as a tuple for L-BFGS-B with jac.
    """
    pts = x_flat.reshape(N, 3)
    pts = _project_to_sphere(pts)
    energy, grad = _log_energy_and_grad(pts)
    # The gradient with respect to the original coordinates needs to account for projection.
    # However, a common trick is to just use the gradient on the sphere and let the optimizer
    # handle the constraint via the parameterization or by projecting after each step.
    # For L-BFGS-B without explicit constraints, we project in the objective.
    # A simpler approach: use the gradient as-is, knowing that the optimizer will step in
    # the flat space, and we project. This is a heuristic but often works.
    # Alternatively, we can use SLSQP with equality constraints, but it's slower.
    # Let's try the projection heuristic first.
    return energy, grad


def construct():
    # Start from a Fibonacci spiral (good initial guess)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Try multiple random restarts to avoid local minima
    best_energy = np.inf
    best_pts = pts_init.copy()
    
    # Define the objective for scipy
    def obj(x):
        e, g = _objective_and_grad(x)
        return e
    
    def grad(x):
        e, g = _objective_and_grad(x)
        return g
    
    # Run optimization with a few restarts
    n_restarts = 5
    
    for restart in range(n_restarts):
        if restart == 0:
            x0 = pts_init.flatten()
        else:
            # Add random perturbation
            angle = np.random.uniform(0, 2 * np.pi, size=(N, 3))
            # Small random rotation
            theta = np.random.uniform(-0.1, 0.1, size=(N, 3))
            pts_perturbed = pts_init + theta
            pts_perturbed = _project_to_sphere(pts_perturbed)
            x0 = pts_perturbed.flatten()
        
        try:
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < best_energy:
                best_energy = result.fun
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
        except Exception:
            continue
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
# EVOLVE-BLOCK-END
