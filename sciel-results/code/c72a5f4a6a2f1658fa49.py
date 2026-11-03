import numpy as np
import time
import math
import random

N = 100

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    return float(-np.sum(np.log(dist[iu])))

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _halton_sphere(dim=3, skip=10):
    """Halton sequence on sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for n in range(skip, skip + N):
        u1 = halton(n, 2)
        u2 = halton(n, 3)
        u3 = halton(n, 5)
        # Use u1, u2 to generate point on sphere
        z = 1.0 - 2.0 * u1
        theta = 2.0 * np.pi * u2
        r = np.sqrt(1.0 - z * z)
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(pts)

def _project(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _optimize(pts, deadline, max_time_per_run=20.0):
    """Riemannian gradient descent with adaptive step size and random kicks."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_energy = _energy(pts)
    
    lr = 0.01
    min_lr = 1e-6
    kick_interval = 50
    kick_count = 0
    
    for t in range(2000):
        if time.time() > deadline or time.time() > start_time + max_time_per_run:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Compute gradient
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = diff / dist_sq[:, :, np.newaxis]
        idx = np.diag_indices(N)
        terms[idx] = 0.0
        grad = -np.sum(terms, axis=1)
        
        # Project gradient onto tangent space
        dot = np.sum(grad * pts, axis=1)[:, np.newaxis]
        grad_proj = grad - dot * pts
        
        # Adaptive step size with backtracking
        new_pts = _project(pts - lr * grad_proj)
        new_energy = _energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
        
        if new_energy < _energy(pts):
            pts = new_pts
            # Decrease learning rate slightly
            lr = max(min_lr, lr * 0.999)
        else:
            # Backtrack: reduce learning rate
            lr = max(min_lr, lr * 0.5)
            if lr <= min_lr:
                # Try a random kick to escape local minimum
                kick_count += 1
                if kick_count % kick_interval == 0:
                    rng = np.random.default_rng()
                    kick = rng.standard_normal((N, 3)) * 0.1
                    pts = _project(pts + kick)
                    lr = 0.01
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Multiple diverse initializations
    initializations = [
        _fibonacci_spiral(),
        _random_sphere(seed=42),
        _random_sphere(seed=123),
        _halton_sphere(),
    ]
    
    best_pts = None
    best_energy = float('inf')
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Allocate remaining time to this run
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Optimize from this initialization
        pts, energy = _optimize(init_pts, deadline, max_time_per_run=min(20.0, remaining))
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    return best_pts
