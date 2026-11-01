"""Rewritten program: multi-start gradient descent with L-BFGS polish for logarithmic energy on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds before deadline to stop


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    n = len(pts)
    # Pairwise distances (chordal)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid self-pairs and very small distances
    np.fill_diagonal(dist, 1.0)
    # Clip to avoid log(0)
    dist = np.maximum(dist, 1e-15)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2 * (-1) ... 
    # Actually: d/dx_i log(1/d) = d/dx_i (-log d) = -1/d * d(d)/dx_i
    # d = |x_i - x_j|, d(d)/dx_i = (x_i - x_j)/d
    # So d/dx_i (-log d) = -(x_i - x_j)/d^2
    # For the full sum over j!=i: grad_i = sum_{j!=i} -(x_i - x_j)/d_ij^2
    # But we need to project onto tangent space of sphere.
    
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d2 = dist[i, j] ** 2
            grad[i] -= (pts[i] - pts[j]) / d2
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) * x_i
    for i in range(n):
        dot = np.dot(grad[i], pts[i])
        grad[i] -= dot * pts[i]
    
    return energy, grad


def _energy_and_grad_fast(pts):
    """Vectorized energy and gradient computation."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(np.maximum(dist2, 1e-30))
    np.fill_diagonal(dist, 1.0)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    # Gradient computation vectorized
    # grad[i] = -sum_{j!=i} (pts[i] - pts[j]) / dist2[i,j]
    inv_dist2 = np.zeros_like(dist2)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2 = 1.0 / np.maximum(dist2, 1e-30)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = -sum_j inv_dist2[i,j] * (pts[i] - pts[j])
    # = -inv_dist2[i,:] dot pts[i] + inv_dist2[i,:] dot pts
    # = -pts[i] * sum_j inv_dist2[i,j] + inv_dist2[i,:] @ pts
    
    sum_inv = np.sum(inv_dist2, axis=1)  # shape (n,)
    grad = -pts * sum_inv[:, np.newaxis] + inv_dist2 @ pts
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1)
    grad -= dot[:, np.newaxis] * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_init():
    """Golden-angle Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(seed):
    """Random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _optimize_single(pts_init, time_limit):
    """Optimize from a single starting point. Returns (energy, points)."""
    start_time = time.time()
    
    # First do L-BFGS-B optimization
    def obj_and_grad(x):
        pts = x.reshape(N, 3)
        e, g = _energy_and_grad_fast(pts)
        return e, g.flatten()
    
    x0 = pts_init.flatten()
    
    result = minimize(
        obj_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 5000, 'maxfun': 5000, 'gtol': 1e-10, 'ftol': 1e-12}
    )
    
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    best_energy, _ = _energy_and_grad_fast(best_pts)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        return best_energy, best_pts
    
    # Polish pass with tighter tolerances
    start_time2 = time.time()
    result2 = minimize(
        obj_and_grad,
        best_pts.flatten(),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 10000, 'maxfun': 10000, 'gtol': 1e-12, 'ftol': 1e-14}
    )
    
    polished_pts = _project_to_sphere(result2.x.reshape(N, 3))
    polished_energy, _ = _energy_and_grad_fast(polished_pts)
    
    if polished_energy < best_energy:
        best_pts = polished_pts
        best_energy = polished_energy
    
    return best_energy, best_pts


def construct():
    """Construct 150 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Generate multiple initializations
    inits = [_fibonacci_init()]
    
    # Add random initializations
    for seed in range(1, 11):
        inits.append(_random_init(seed))
    
    # Add perturbed fibonacci
    base = _fibonacci_init()
    for k in range(5):
        rng = np.random.RandomState(100 + k)
        perturbed = base + 0.1 * rng.randn(N, 3)
        inits.append(_project_to_sphere(perturbed))
    
    best_energy = float('inf')
    best_pts = inits[0]
    
    for idx, init in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - SAFETY_MARGIN - elapsed
        if remaining <= 5:  # Not enough time for another optimization
            break
        
        try:
            energy, pts = _optimize_single(init, remaining - 10)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception:
            continue
    
    # Final check: ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] = best_pts[i] + 1e-8 * best_pts[j]
                best_pts[j] = _project_to_sphere(best_pts[j])
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_init()
    
    return best_pts
