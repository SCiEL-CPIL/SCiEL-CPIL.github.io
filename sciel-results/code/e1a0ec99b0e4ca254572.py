"""Rewritten program: multi-restart basin hopping with CG on sphere, log-energy minimization."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = 145.0  # safety margin before 150s kill

def _log_energy(pts):
    """Compute log(1/d) energy for all pairs using math.fsum."""
    d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, 1)
    return math.fsum(np.log(1.0 / dist[iu]).tolist())

def _energy_and_grad(pts):
    """Return (energy, gradient) for E = sum_{i<j} -log(|xi-xj|)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    weight = -1.0 / dist_sq
    np.fill_diagonal(weight, 0.0)
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    iu = np.triu_indices(n, 1)
    energy = math.fsum(np.log(1.0 / dist[iu]).tolist())
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(seed_offset=0.0):
    i = np.arange(N) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    pts = rng.standard_normal((N, 3))
    return _project_to_sphere(pts)

def _cg_on_sphere(pts, max_iters=5000, tol=1e-10, max_time=30.0):
    """Conjugate gradient method on the sphere with line search."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    n = N
    t0 = time.time()
    
    def obj_and_grad(x):
        p = _project_to_sphere(x)
        e, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tan = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tan
    
    best_energy, best_pts = _log_energy(pts), pts.copy()
    
    for it in range(max_iters):
        if time.time() - t0 > max_time:
            break
        
        e, g = obj_and_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        if np.linalg.norm(g) < tol:
            break
        
        if it == 0:
            d = -g
        else:
            beta = np.dot(g, g) / max(np.dot(g_prev, g_prev), 1e-30)
            d = -g + beta * d
            # Ensure descent direction
            if np.dot(d, g) > 0:
                d = -g
        g_prev = g
        
        # Line search (backtracking)
        alpha = 0.01
        for _ in range(50):
            new_pts = _project_to_sphere(pts + alpha * d)
            new_e = _log_energy(new_pts)
            if new_e < e - 1e-4 * alpha * np.dot(g, d):
                pts = new_pts
                break
            alpha *= 0.5
        else:
            break
    
    return best_pts, best_energy

def _local_minimize(pts, max_time=20.0):
    """Local minimization using CG, then a final polish."""
    pts, e = _cg_on_sphere(pts, max_iters=3000, tol=1e-10, max_time=max_time * 0.7)
    
    # Final polish with very small steps
    t0 = time.time()
    lr = 1e-5
    iu = np.triu_indices(N, 1)
    for _ in range(2000):
        if time.time() - t0 > max_time * 0.3:
            break
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        weight = -1.0 / dist_sq
        np.fill_diagonal(weight, 0.0)
        grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        new_pts = _project_to_sphere(pts - lr * grad_tan)
        diff2 = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :]
        dist2 = np.sqrt(np.sum(diff2 * diff2, axis=2))
        np.fill_diagonal(dist2, 1.0)
        e2 = -np.log(dist2[iu]).sum()
        
        if e2 < e - 1e-14:
            pts = new_pts
            e = e2
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return pts, _log_energy(pts)

def _basin_hopping_single(pts, rng, max_time=15.0):
    """Single basin hopping trajectory."""
    t0 = time.time()
    best_pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    best_energy = _log_energy(best_pts)
    
    temp = 0.05
    for step in range(50):
        if time.time() - t0 > max_time:
            break
        
        # Perturbation
        perturbation = 0.1 * rng.standard_normal((N, 3))
        perturbed = _project_to_sphere(best_pts + perturbation)
        
        # Local minimize the perturbed point
        candidate, e_cand = _local_minimize(perturbed, max_time=max(1.0, (max_time - (time.time() - t0)) * 0.5))
        
        # Metropolis acceptance
        delta_e = e_cand - best_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            best_pts = candidate
            best_energy = e_cand
            temp *= 0.9
        else:
            temp *= 0.95
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Phase 1: Multi-restart local minimization from diverse starts
    best_energy = float('inf')
    best_pts = None
    
    # Start 1: Fibonacci
    pts = _fibonacci_sphere()
    pts, e = _local_minimize(pts, max_time=25.0)
    best_energy, best_pts = e, pts
    
    # Start 2: Rotated Fibonacci
    remaining = _DEADLINE - time.time()
    if remaining > 20:
        pts = _fibonacci_sphere(seed_offset=0.25)
        # Apply random rotation
        theta = rng.standard_normal(3)
        u, v, w = theta / np.linalg.norm(theta)
        c, s = np.cos(1.0), np.sin(1.0)
        R = np.array([
            [c + u*u*(1-c), u*v*(1-c) - w*s, u*w*(1-c) + v*s],
            [v*u*(1-c) + w*s, c + v*v*(1-c), v*w*(1-c) - u*s],
            [w*u*(1-c) - v*s, w*v*(1-c) + u*s, c + w*w*(1-c)]
        ])
        pts = _project_to_sphere(pts @ R.T)
        pts, e = _local_minimize(pts, max_time=20.0)
        if e < best_energy:
            best_energy, best_pts = e, pts
    
    # Start 3: Random
    remaining = _DEADLINE - time.time()
    if remaining > 20:
        pts = _random_sphere(rng)
        pts, e = _local_minimize(pts, max_time=15.0)
        if e < best_energy:
            best_energy, best_pts = e, pts
    
    # Phase 2: Basin hopping from best configuration
    remaining = _DEADLINE - time.time()
    if remaining > 25:
        pts, e = _basin_hopping_single(best_pts, rng, max_time=min(remaining - 5, 40.0))
        if e < best_energy:
            best_energy, best_pts = e, pts
    
    # Phase 3: Final high-precision polish
    remaining = _DEADLINE - time.time()
    if remaining > 3:
        best_pts, best_energy = _local_minimize(best_pts, max_time=remaining - 2.0)
    
    # Ensure valid output
    final = _project_to_sphere(np.asarray(best_pts, dtype=np.float64))
    # Verify no points too close
    d = final[:, np.newaxis, :] - final[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fall back to Fibonacci
        final = _fibonacci_sphere()
    
    return final
# EVOLVE-BLOCK-END
