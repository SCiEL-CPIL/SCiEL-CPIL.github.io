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

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Initializations
    inits = [_golden_spiral(), _greedy_placement()]
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        inits.append(_random_sphere_points(seed))
        
    def objective_with_grad(x):
        pts_candidate = x.reshape(N, 3)
        energy, grad = _fast_energy_grad(pts_candidate)
        return energy, grad.flatten()
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            # First pass
            result = minimize(
                objective_with_grad,
                init_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
            )
            
            candidate_pts = _normalize(result.x.reshape(N, 3))
            candidate_energy = _log_energy(candidate_pts)
            
            if candidate_energy < best_energy and np.all(np.isfinite(candidate_pts)):
                best_pts = candidate_pts.copy()
                best_energy = candidate_energy
                
                # Second pass: refinement
                if time.time() < deadline - 15:
                    result2 = minimize(
                        objective_with_grad,
                        best_pts.flatten(),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000}
                    )
                    
                    candidate_pts2 = _normalize(result2.x.reshape(N, 3))
                    candidate_energy2 = _log_energy(candidate_pts2)
                    
                    if candidate_energy2 < best_energy and np.all(np.isfinite(candidate_pts2)):
                        best_pts = candidate_pts2.copy()
                        best_energy = candidate_energy2
                        
        except Exception:
            continue
            
    if best_pts is None:
        best_pts = _golden_spiral()
        
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _golden_spiral()
        
    return best_pts
