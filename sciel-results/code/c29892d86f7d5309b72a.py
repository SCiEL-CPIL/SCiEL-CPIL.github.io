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

def _fast_energy_and_grad(pts):
    """
    Vectorized energy and gradient computation for optimization.
    Returns (energy, gradient) where gradient is shape (N, 3).
    
    Energy: E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_k: grad_k = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    """
    n = pts.shape[0]
    # Compute pairwise differences: diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    # Squared distances
    sq_dists = np.sum(diff**2, axis=2)  # (n, n)
    # Clip to avoid division by zero
    sq_dists = np.maximum(sq_dists, 1e-24)
    dists = np.sqrt(sq_dists)  # (n, n)
    
    # Energy: sum over i<j of log(1/d_ij) = -sum over i<j of log(d_ij)
    # Use upper triangle excluding diagonal
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each point k, grad_k = -sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # diff[i,j] = x_i - x_j, so grad_i = -sum_{j != i} diff[i,j] / sq_dists[i,j]
    # We need to be careful: the gradient of -log(|x_i - x_j|) w.r.t. x_i is -(x_i - x_j)/|x_i-x_j|^2
    # So grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2 * (-1) = -sum_{j != k} (x_k - x_j)/|x_k-x_j|^2
    
    # Compute gradient contributions
    # grad[i] = -sum_{j != i} diff[i,j] / sq_dists[i,j]
    # diff[i,j] / sq_dists[i,j] has shape (n, n, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_diff = diff / sq_dists[:, :, np.newaxis]  # (n, n, 3)
    # Set diagonal to 0
    weighted_diff[np.arange(n), np.arange(n), :] = 0.0
    # Gradient for each point i: -sum over j of weighted_diff[i,j]
    grad = -np.sum(weighted_diff, axis=1)  # (n, 3)
    
    # Project gradient to tangent space: grad_tangent = grad - (grad . x_k) x_k
    # Since points are on unit sphere, tangent space is orthogonal to x_k
    dot_products = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tangent = grad - dot_products * pts  # (n, 3)
    
    return energy, grad_tangent

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    return _normalize(pts)

def _golden_spiral():
    """Generate golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    Uses golden-angle spiral as initial guess, then optimizes with L-BFGS-B
    using analytical gradients projected to the tangent space.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate initial points
    pts = _golden_spiral()
    
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Objective function for optimization (returns scalar energy)
    def objective(x):
        pts_candidate = _project_to_sphere(x.reshape(N, 3))
        energy, _ = _fast_energy_and_grad(pts_candidate)
        return energy
    
    # Gradient function (returns flat gradient)
    def gradient(x):
        pts_candidate = _project_to_sphere(x.reshape(N, 3))
        _, grad = _fast_energy_and_grad(pts_candidate)
        return grad.flatten()
    
    try:
        # First optimization pass
        result = minimize(
            objective,
            pts.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        if result.fun < best_energy:
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts
                best_energy = candidate_energy
        
        # Check if we have time for a second refinement pass
        elapsed = time.time() - start_time
        if elapsed < deadline - 40:
            # Second optimization pass with tighter tolerances
            result2 = minimize(
                objective,
                best_pts.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            
            if result2.fun < best_energy:
                candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                candidate_energy2 = _log_energy(candidate_pts2)
                if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                    best_pts = candidate_pts2
                    best_energy = candidate_energy2
        
        # Final high-precision polish pass if time permits
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            # Use a very tight tolerance for final polish
            result3 = minimize(
                objective,
                best_pts.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 20000}
            )
            
            if result3.fun < best_energy:
                candidate_pts3 = _normalize(result3.x.reshape(N, 3))
                candidate_energy3 = _log_energy(candidate_pts3)
                if candidate_energy3 < best_energy and np.all(np.isfinite(candidate_pts3)):
                    best_pts = candidate_pts3
                    best_energy = candidate_energy3
                    
    except Exception as e:
        # If optimization fails, keep the best result so far
        pass
    
    # Final validation
    best_pts = _normalize(best_pts)
    
    # Check for validity
    if not np.all(np.isfinite(best_pts)):
        # Fallback to golden spiral
        best_pts = _golden_spiral()
    
    return best_pts
# EVOLVE-BLOCK-END
