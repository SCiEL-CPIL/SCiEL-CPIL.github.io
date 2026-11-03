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
    """Vectorized energy computation for optimization."""
    n = pts.shape[0]
    sq_dists = np.sum(pts**2, axis=1)[:, None] + np.sum(pts**2, axis=1)[None, :] - 2 * pts @ pts.T
    sq_dists = np.maximum(sq_dists, 0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(sq_dists[iu])
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
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = pts / norms  # projected points on sphere
    
    # Pairwise distances
    sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2 * x @ x.T
    sq_dists = np.maximum(sq_dists, 0)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, 1.0)  # avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient of E w.r.t. x: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    # Because dE/dx_i = -sum_{j!=i} d/dx_i log(d_ij) = -sum_{j!=i} (x_i-x_j)/(d_ij^2 * d_ij) * d_ij = sum (x_i-x_j)/d_ij^2
    # Actually: d/dx_i log(d_ij) = (x_i - x_j) / d_ij^2
    # So dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2 = sum_{j!=i} (x_j - x_i)/d_ij^2
    
    # Vectorized gradient computation
    # For each pair (i,j), contribution to grad[i] is (x_j - x_i)/d_ij^2, and to grad[j] is (x_i - x_j)/d_ij^2
    grad_x = np.zeros_like(x)
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if d < 1e-12:
                continue
            diff = x[i] - x[j]
            grad_x[i] += diff / (d * d)
            grad_x[j] -= diff / (d * d)
    
    # Chain rule: x_i = p_i / ||p_i||
    # dx_i/dp_i = (I - x_i x_i^T) / ||p_i||
    # So dE/dp_i = (I - x_i x_i^T) @ grad_x[i] / ||p_i||
    grad_p = np.zeros_like(pts)
    for i in range(n):
        norm_i = norms[i, 0]
        if norm_i < 1e-12:
            continue
        # Projection: (I - x_i x_i^T) @ grad_x[i]
        proj_grad = grad_x[i] - np.dot(x[i], grad_x[i]) * x[i]
        grad_p[i] = proj_grad / norm_i
    
    return energy, grad_p

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize(pts)

def _golden_spiral():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _random_sphere_points(seed):
    """Generate random points on unit sphere with given seed."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _normalize(pts)

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses multiple initializations with L-BFGS-B and analytical gradients.
    """
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # 1. Golden spiral
    initializations.append(_golden_spiral())
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        initializations.append(_random_sphere_points(seed))
    
    def objective_with_grad(x):
        """Objective and gradient for L-BFGS-B."""
        pts_candidate = x.reshape(N, 3)
        energy, grad = _fast_energy_grad(pts_candidate)
        return energy, grad.flatten()
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        # Skip if we already have a very good solution and this is a random init
        if best_energy < -1083.0 and init_idx > 0 and init_idx != 1:
            # Only try golden spiral and first random if we have a great result
            if init_idx > 2:
                continue
        
        try:
            # First pass: moderate optimization
            result = minimize(
                objective_with_grad,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 3000}
            )
            
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts.copy()
                best_energy = candidate_energy
                
                # Second pass: fine refinement if we have time
                if time.time() < deadline - 20:
                    result2 = minimize(
                        objective_with_grad,
                        best_pts.flatten(),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
                    )
                    
                    candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                    candidate_energy2 = _log_energy(candidate_pts2)
                    
                    if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                        best_pts = candidate_pts2.copy()
                        best_energy = candidate_energy2
                        
        except Exception:
            continue
    
    # If no valid result, fall back to golden spiral
    if best_pts is None:
        best_pts = _golden_spiral()
    
    # Final validation
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _golden_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
