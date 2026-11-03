"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fast_energy(pts):
    """Vectorized energy computation for optimization (less precise but faster)."""
    n = pts.shape[0]
    # Compute pairwise squared distances
    sq_dists = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
    sq_dists = np.maximum(sq_dists, 0)
    # Only upper triangle, exclude diagonal
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(sq_dists[iu])
    # Clip to avoid log(0)
    dists = np.maximum(dists, 1e-12)
    return np.sum(np.log(1.0 / dists))

def _fast_energy_grad(pts):
    """
    Compute energy and its gradient w.r.t. pts (before projection).
    Energy = sum_{i<j} log(1/|x_i - x_j|) where x_i are projected to sphere.
    We compute dE/dx_i for the projected points, then chain rule for the raw coords.
    """
    n = pts.shape[0]
    # Project to sphere
    x = _normalize(pts)
    
    # Compute pairwise distances
    sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2 * x @ x.T
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, 1.0)  # avoid division by zero on diagonal
    
    # Gradient of E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    # Wait: d/dx log(1/r) = d/dx (-log r) = -1/r * dr/dx = -1/r * (x_i - x_j)/r = -(x_i - x_j)/r^2
    # So dE/dx_i = sum_{j != i} -(x_i - x_j)/|x_i - x_j|^2
    # But we want to minimize E, so gradient points in direction of increasing E.
    # For minimization, we want to move in -gradient direction.
    
    # Let's compute: grad[i] = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Actually, let me re-derive:
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} d/dx_i log(d_ij) - sum_{j<i} d/dx_i log(d_ij)
    # d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / d_ij^2
    
    # Vectorized computation
    # diff[i,j] = x_i - x_j
    diff = x[:, None, :] - x[None, :, :]  # shape (n, n, 3)
    dists_sq = np.sum(diff**2, axis=2)  # shape (n, n)
    np.fill_diagonal(dists_sq, 1.0)
    
    # weight[i,j] = 1/dists_sq[i,j]
    weights = 1.0 / dists_sq  # shape (n, n)
    
    # grad[i] = -sum_j weights[i,j] * diff[i,j]
    # grad = -sum over j of (1/d^2) * (x_i - x_j)
    grad_on_sphere = -np.sum(weights[:, :, None] * diff, axis=1)  # shape (n, 3)
    
    # Now chain rule: x = pts / |pts|
    # dx/dpts = (I - x x^T) / |pts|  (projection onto tangent space divided by norm)
    # Actually, for normalization: x = pts / ||pts||
    # dx_i/dpts_j = (delta_ij - x_i x_j) / ||pts||
    # So dE/dpts = sum_i (dE/dx_i) * (I - x_i x_i^T) / ||pts_i||
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)  # shape (n, 1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    # dE/dpts = (I - x x^T) * grad_on_sphere / norms
    # (I - x x^T) @ grad_on_sphere = grad_on_sphere - x * (x @ grad_on_sphere)
    x_dot_grad = np.sum(x * grad_on_sphere, axis=1, keepdims=True)  # shape (n, 1)
    projected_grad = grad_on_sphere - x * x_dot_grad
    grad = projected_grad / norms
    
    energy = _fast_energy(x)
    return energy, grad

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize(pts)

def _golden_spiral():
    """Generate golden-angle spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_sphere_points(n, seed=None):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses golden-angle spiral as initial guess, then optimizes with L-BFGS-B using analytical gradient.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good starting point.
    pts = _golden_spiral()
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Optimization using L-BFGS-B on the sphere with analytical gradient
    
    def objective_and_grad(x):
        pts_candidate = x.reshape(N, 3)
        energy, grad = _fast_energy_grad(pts_candidate)
        return energy, grad.flatten()
    
    def objective(x):
        pts_candidate = _project_to_sphere(x.reshape(N, 3))
        return _fast_energy(pts_candidate)
    
    try:
        # Optimize with L-BFGS-B using analytical gradient
        result = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        
        if result.fun < best_energy:
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts
                best_energy = candidate_energy
        
        # Check if we have time for a second refinement pass
        if time.time() < deadline - 30:
            # Use the optimized points as new starting point for a finer optimization
            result2 = minimize(
                objective_and_grad,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 20000}
            )
            
            if result2.fun < best_energy:
                candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                candidate_energy2 = _log_energy(candidate_pts2)
                if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                    best_pts = candidate_pts2
                    best_energy = candidate_energy2
                    
    except Exception as e:
        # If optimization fails, return the initial golden spiral
        pass
    
    # Final validation
    best_pts = _normalize(best_pts)
    
    # Check for validity
    if not np.all(np.isfinite(best_pts)):
        # Fallback to golden spiral
        best_pts = _golden_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
