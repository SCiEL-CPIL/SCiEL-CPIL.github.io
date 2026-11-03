"""Rewritten program: cooperative multipoint gradient descent with multi-starts and polish."""
import numpy as np
import math
import time

N = 100

def _log_energy_and_grad(pts):
    """Compute log energy and gradient for points on unit sphere (chordal distances).
    Energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient w.r.t. xi for the energy (to minimize):
      d/dxi [ -log(|xi-xj|) ] = (xi-xj)/|xi-xj|^2  (for each j != i)
    But we want to minimize energy, so gradient of energy is:
      grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    We project gradient to tangent space: grad_tan = grad - (grad . xi) * xi
    """
    # pts: (N, 3)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (N, N)
    
    # Avoid division by zero on diagonal
    dist_safe = np.where(np.eye(N, dtype=bool), 1.0, dist)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist_safe[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (xi - xj) / dist_ij^2
    # diff[i,j] = xi - xj
    # We need (xi - xj) / dist_ij^2 for each pair
    # inv_dist_sq: (N, N)
    inv_dist_sq = np.zeros_like(dist_safe)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist_safe[mask] ** 2)
    
    # grad_i = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)  # (N, 1)
    grad_tan = grad - dot * pts  # (N, 3)
    
    return energy, grad_tan


def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, tol=1e-12, time_limit=None, start_time=None):
    """Cooperative gradient descent with adaptive step size."""
    if start_time is None:
        start_time = time.time()
    
    pts = _project_to_sphere(pts.copy())
    energy, grad_tan = _log_energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    lr = lr_init
    min_lr = 1e-6
    max_lr = 0.1
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Adaptive step: use gradient norm to scale
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-14:
            break
        
        # Take step in negative gradient direction
        step = lr * grad_tan
        new_pts = pts - step
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, new_grad_tan = _log_energy_and_grad(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            grad_tan = new_grad_tan
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            # Increase learning rate slightly
            lr = min(lr * 1.05, max_lr)
        else:
            # Decrease learning rate
            lr = max(lr * 0.7, min_lr)
            if lr <= min_lr * 1.01:
                break
        
        if abs(energy - best_energy) < tol and it > 100:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Multi-start: golden spiral + random starts
    starts = [_golden_spiral()]
    
    # Add several random starts
    n_random = 8
    for _ in range(n_random):
        starts.append(_random_start(rng))
    
    best_pts = None
    best_energy = np.inf
    
    for start in starts:
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        # Allocate remaining time proportionally
        remaining = deadline - elapsed
        time_per_start = min(remaining / max(1, len(starts) - len(starts[:starts.index(start)])), 20)
        
        pts, energy = _gradient_descent(
            start, 
            max_iter=3000, 
            lr_init=0.005, 
            tol=1e-13,
            time_limit=time_per_start,
            start_time=start_time
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish: tighter gradient descent on the best point
    if best_pts is not None and time.time() < deadline - 2:
        remaining_time = deadline - time.time()
        best_pts, best_energy = _gradient_descent(
            best_pts,
            max_iter=10000,
            lr_init=0.001,
            tol=1e-15,
            time_limit=remaining_time,
            start_time=start_time
        )
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts = _project_to_sphere(best_pts)
                break
    
    return best_pts
