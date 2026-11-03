"""Rewritten program with vectorized gradient and corrected sign for minimization."""
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

def _fast_energy_grad(pts):
    """
    Compute energy and its gradient w.r.t. pts (before projection).
    Energy = sum_{i<j} log(1/|x_i - x_j|).
    Returns (energy, grad_p) where grad_p is the gradient w.r.t. the input points.
    """
    n = pts.shape[0]
    # Project to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x = pts / norms  # projected points on sphere
    
    # Pairwise squared distances
    sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2 * (x @ x.T)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / np.sqrt(sq_dists[iu])))
    
    # Gradient of E w.r.t x
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # inv_d2[i,j] = 1 / d_ij^2
    inv_d2 = 1.0 / sq_dists
    np.fill_diagonal(inv_d2, 0.0)
    
    # diff[i,j] = x_i - x_j
    diff = x[:, None, :] - x[None, :, :]
    
    # grad_x[i] = sum_{j!=i} (x_j - x_i) * inv_d2[i,j] = - sum_{j!=i} (x_i - x_j) * inv_d2[i,j]
    weighted_diff = diff * inv_d2[..., None]
    grad_x = -np.sum(weighted_diff, axis=1)
    
    # Chain rule: x_i = p_i / ||p_i||
    # dx_i/dp_i = (I - x_i x_i^T) / ||p_i||
    # dE/dp_i = (I - x_i x_i^T) @ grad_x[i] / ||p_i||
    grad_p = np.zeros_like(pts)
    for i in range(n):
        norm_i = norms[i, 0]
        if norm_i < 1e-12:
            continue
        proj_grad = grad_x[i] - np.dot(x[i], grad_x[i]) * x[i]
        grad_p[i] = proj_grad / norm_i
    
    return energy, grad_p

def _normalize(pts):
    """Normalize points to unit sphere."""
    if pts.ndim == 1:
        norm = np.linalg.norm(pts)
        if norm < 1e-12:
            return pts.copy()
        return pts / norm
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

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

def _greedy_placement():
    """Greedy placement: place points one by one maximizing min distance."""
    pts = np.zeros((N, 3))
    # Start with a random point
    rng = np.random.RandomState(0)
    pts[0] = _normalize(rng.randn(3))
    
    for k in range(1, N):
        best_min_dist = -1
        best_candidate = None
        # Try many random candidates
        for _ in range(1000):
            cand = _normalize(rng.randn(3))
            # Compute min distance to existing points
            dists = np.linalg.norm(pts[:k] - cand, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = cand
        pts[k] = best_candidate
    return _normalize(pts)

def _sobol_sphere_points():
    """Generate points on the sphere using a quasi-random (Sobol-like) sequence."""
    # Use a simple low-discrepancy sequence based on golden ratio in 3D
    # For better coverage, use Halton-like sequences
    n = N
    pts = np.zeros((n, 3))
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    for i in range(n):
        idx = i + 1
        u1 = halton(idx, 2)
        u2 = halton(idx, 3)
        u3 = halton(idx, 5)
        
        # Map to sphere using inverse CDF for uniform distribution
        z = 2.0 * u1 - 1.0  # [-1, 1]
        theta = 2.0 * np.pi * u2
        r = np.sqrt(1.0 - z * z)
        pts[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
    
    return _normalize(pts)

def _sphere_energy_and_grad_sphere(x):
    """
    Compute energy and gradient for points already on the unit sphere.
    This avoids the chain rule through normalization.
    x: (N, 3) array of points on the unit sphere
    Returns (energy, grad) where grad is the gradient projected onto the tangent space.
    """
    n = x.shape[0]
    
    # Pairwise squared distances
    sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(x**2, axis=1)[None, :] - 2 * (x @ x.T)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    energy = np.sum(np.log(1.0 / np.sqrt(sq_dists[iu])))
    
    # Gradient of E w.r.t x (in R^3)
    inv_d2 = 1.0 / sq_dists
    np.fill_diagonal(inv_d2, 0.0)
    
    # diff[i,j] = x_i - x_j
    diff = x[:, None, :] - x[None, :, :]
    
    # grad_x[i] = sum_{j!=i} (x_j - x_i) * inv_d2[i,j] = - sum_{j!=i} (x_i - x_j) * inv_d2[i,j]
    weighted_diff = diff * inv_d2[..., None]
    grad_x = -np.sum(weighted_diff, axis=1)
    
    # Project gradient onto tangent space: grad_tangent[i] = grad_x[i] - (x_i . grad_x[i]) * x_i
    dot_products = np.sum(x * grad_x, axis=1, keepdims=True)
    grad_tangent = grad_x - dot_products * x
    
    return energy, grad_tangent

def _tangent_space_optimize(pts, maxiter=1000, tol=1e-12):
    """
    Optimize points on the sphere using gradient descent in tangent space.
    Uses adaptive step size with line search for better convergence.
    """
    n = pts.shape[0]
    current_pts = _normalize(pts.copy())
    current_energy = _log_energy(current_pts)
    
    # Initial step size estimate based on gradient magnitude
    _, grad_tangent = _sphere_energy_and_grad_sphere(current_pts)
    grad_norm = np.linalg.norm(grad_tangent)
    if grad_norm > 0:
        step = min(0.5, 1.0 / (grad_norm + 1e-10))
    else:
        step = 0.1
    
    for it in range(maxiter):
        energy, grad_tangent = _sphere_energy_and_grad_sphere(current_pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < tol:
            break
        
        # Line search with backtracking
        best_step = step
        best_energy = current_energy
        best_pts = current_pts
        
        for ls in range(20):
            candidate_pts = current_pts - grad_tangent * best_step
            candidate_pts = _normalize(candidate_pts)
            candidate_energy = _log_energy(candidate_pts)
            
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_pts = candidate_pts.copy()
                # Increase step if we made good progress
                if ls > 0:
                    step = best_step * 1.5
                break
            else:
                best_step *= 0.5
        
        if best_energy < current_energy:
            current_pts = best_pts
            current_energy = best_energy
        else:
            # If no improvement, reduce step and continue
            step *= 0.5
            if step < 1e-15:
                break
    
    return current_pts, current_energy

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer before 150s kill)
    
    best_pts = None
    best_energy = float('inf')
    
    # Initializations: diverse starting points
    inits = [_golden_spiral(), _greedy_placement(), _sobol_sphere_points()]
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222]:
        inits.append(_random_sphere_points(seed))
        
    def objective_with_grad(x):
        pts_candidate = x.reshape(N, 3)
        energy, grad = _fast_energy_grad(pts_candidate)
        return energy, grad.flatten()
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25:
            break
            
        try:
            # First pass: L-BFGS-B with moderate tolerance
            result = minimize(
                objective_with_grad,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-9, 'maxfun': 4000}
            )
            
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts.copy()
                best_energy = candidate_energy
                
                # Second pass: tighter L-BFGS-B
                if time.time() < deadline - 35:
                    result2 = minimize(
                        objective_with_grad,
                        best_pts.flatten(),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-11, 'maxfun': 6000}
                    )
                    
                    candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                    candidate_energy2 = _log_energy(candidate_pts2)
                    
                    if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                        best_pts = candidate_pts2.copy()
                        best_energy = candidate_energy2
                        
                # Third pass: tangent space gradient descent for final polish
                if time.time() < deadline - 15 and best_pts is not None:
                    polished_pts, polished_energy = _tangent_space_optimize(
                        best_pts, maxiter=3000, tol=1e-14
                    )
                    
                    if polished_energy < best_energy and np.all(np.isfinite(polished_pts)):
                        best_pts = polished_pts.copy()
                        best_energy = polished_energy
                        
                    # Fourth pass: ultra-tight polish if time permits
                    if time.time() < deadline - 5 and best_pts is not None:
                        polished_pts2, polished_energy2 = _tangent_space_optimize(
                            best_pts, maxiter=5000, tol=1e-15
                        )
                        
                        if polished_energy2 < best_energy and np.all(np.isfinite(polished_pts2)):
                            best_pts = polished_pts2.copy()
                            best_energy = polished_energy2
                        
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = _golden_spiral()
        
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _golden_spiral()
        
    return best_pts
