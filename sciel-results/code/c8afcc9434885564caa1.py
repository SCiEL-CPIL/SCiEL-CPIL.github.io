"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 140.0  # safety margin before 150s kill

def _log_energy(pts):
    """Compute log(1/d) energy for all pairs."""
    # pts: (N,3)
    d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    # Avoid self-pairs and division by zero
    np.fill_diagonal(dist, 1.0)
    # Only upper triangle
    iu = np.triu_indices(N, 1)
    return math.fsum(np.log(1.0 / dist[iu]).tolist())

def _energy_and_grad(pts):
    """Return (energy, gradient) for log(1/|xi-xj|) energy.
    E = sum_{i<j} -log(|xi-xj|)
    dE/dxi = -sum_{j!=i} (xi - xj) / |xi - xj|^2  (for each pair, factor 1/2 handled)
    Actually: d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    So grad_i = -sum_{j!=i} (xi - xj)/|xi-xj|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n,n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by 0
    dist = np.sqrt(dist_sq)
    
    # Gradient: grad_i = -sum_{j!=i} (xi-xj) / |xi-xj|^2
    # weight[i,j] = -1 / |xi-xj|^2
    weight = -1.0 / dist_sq  # (n,n)
    np.fill_diagonal(weight, 0.0)
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)  # (n,3)
    
    # Energy
    iu = np.triu_indices(n, 1)
    energy = math.fsum(np.log(1.0 / dist[iu]).tolist())
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent(pts, max_time, lr_init=0.01):
    """Riemannian gradient descent on the sphere with adaptive step size."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    lr = lr_init
    t0 = time.time()
    
    # Precompute indices for speed
    n = N
    iu = np.triu_indices(n, 1)
    
    step = 0
    while time.time() - t0 < max_time:
        # Compute gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        weight = -1.0 / dist_sq
        np.fill_diagonal(weight, 0.0)
        grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adaptive step: try to reduce energy
        for attempt in range(3):
            new_pts = pts - lr * grad_tan
            new_pts = _project_to_sphere(new_pts)
            
            # Quick energy check
            diff2 = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :]
            dist2 = np.sqrt(np.sum(diff2 * diff2, axis=2))
            np.fill_diagonal(dist2, 1.0)
            e2 = -np.log(dist2[iu]).sum()
            
            if e2 < best_energy - 1e-12:
                pts = new_pts
                best_energy = e2
                best_pts = pts.copy()
                lr = min(lr * 1.05, 0.5)  # increase step slightly
                break
            else:
                lr *= 0.5
        else:
            # Couldn't improve; try smaller step or break
            if lr < 1e-10:
                break
            lr = 1e-10
        
        step += 1
        if step % 100 == 0:
            # Recompute exact energy for tracking
            pass
    
    return best_pts, best_energy

def _polish(pts, max_time):
    """High-precision final polish with very small steps."""
    pts = _project_to_sphere(np.array(pts, dtype=np.float64))
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    t0 = time.time()
    lr = 1e-4
    n = N
    iu = np.triu_indices(n, 1)
    
    while time.time() - t0 < max_time:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        weight = -1.0 / dist_sq
        np.fill_diagonal(weight, 0.0)
        grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
        grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        new_pts = pts - lr * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        diff2 = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :]
        dist2 = np.sqrt(np.sum(diff2 * dist2, axis=2))
        np.fill_diagonal(dist2, 1.0)
        e2 = -np.log(dist2[iu]).sum()
        
        if e2 < best_energy - 1e-14:
            pts = new_pts
            best_energy = e2
            best_pts = pts.copy()
        else:
            lr *= 0.7
            if lr < 1e-12:
                break
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    remaining = _DEADLINE - t_start
    
    # Phase 1: Fibonacci start
    pts = _fibonacci_sphere()
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Phase 2: Gradient descent (main optimization)
    gd_time = remaining * 0.6
    if gd_time > 5:
        pts, e = _gradient_descent(pts, gd_time, lr_init=0.005)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Phase 3: Polish
    remaining = _DEADLINE - time.time()
    if remaining > 2:
        pts, e = _polish(best_pts, remaining - 1.0)
        if e < best_energy:
            best_energy = e
            best_pts = pts
    
    # Final check
    final = _project_to_sphere(best_pts)
    return final
# EVOLVE-BLOCK-END
