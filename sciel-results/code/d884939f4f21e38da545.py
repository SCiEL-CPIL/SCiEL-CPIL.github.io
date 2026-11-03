"""Refined logarithmic energy minimizer for 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100
DEADLINE = 140.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_k: sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    (This is the Euclidean gradient; we project to tangent space in the optimizer.)
    """
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_k = sum_{j != k} (x_k - x_j) / dist_kj^2
    # diff[k,j] = x_k - x_j
    # grad[k] = sum_j diff[k,j] / dist[k,j]^2
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2 = 1.0 / (dist * dist)
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_tangent(pts, grad):
    """Project Euclidean gradient to tangent space of the sphere."""
    # Tangent projection: g - (g . x) x
    dot = np.sum(grad * pts, axis=1)
    return grad - dot[:, np.newaxis] * pts


def _normalize(pts):
    """Re-normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-15] = 1.0
    return pts / norms[:, np.newaxis]


def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent(pts, max_iter, deadline, lr_init=0.01, lr_decay=0.999):
    """Riemannian gradient descent with adaptive step size and re-normalization."""
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    lr = lr_init
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        _, grad = _energy_and_grad(pts)
        tgrad = _project_tangent(pts, grad)
        
        # Adaptive step: if energy increases, halve step
        step = lr
        for _ in range(10):
            new_pts = _normalize(pts - step * tgrad)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy or step < 1e-15:
                break
            step *= 0.5
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            # Slowly grow step if we've been improving
            lr = min(lr * 1.01, lr_init)
        else:
            lr *= 0.5
        
        if lr < 1e-12:
            break
    
    return best_pts, best_energy


def _high_precision_polish(pts, deadline, max_iter=5000):
    """Final polish with very small steps for high-precision convergence."""
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    lr = 1e-4
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        _, grad = _energy_and_grad(pts)
        tgrad = _project_tangent(pts, grad)
        
        # Very careful line search
        step = lr
        improved = False
        for _ in range(8):
            new_pts = _normalize(pts - step * tgrad)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-15:
                pts = new_pts
                energy = new_energy
                improved = True
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
                break
            step *= 0.5
        
        if not improved:
            lr *= 0.7
            if lr < 1e-18:
                break
        else:
            lr = min(lr * 1.05, 1e-3)
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    
    # Start from Fibonacci spiral
    pts = _fibonacci_sphere(N)
    
    # Phase 1: Coarse gradient descent
    deadline1 = start_time + 60.0
    pts, energy = _gradient_descent(pts, max_iter=2000, deadline=deadline1, lr_init=0.05, lr_decay=0.999)
    
    # Phase 2: Medium refinement
    deadline2 = start_time + 100.0
    pts, energy = _gradient_descent(pts, max_iter=3000, deadline=deadline2, lr_init=0.01, lr_decay=0.999)
    
    # Phase 3: High-precision polish
    deadline3 = DEADLINE - 5.0
    pts, energy = _high_precision_polish(pts, deadline=deadline3, max_iter=10000)
    
    # Final normalization and validation
    pts = _normalize(pts)
    
    # Validate: check finite values, nonzero, pairwise distances
    if not np.all(np.isfinite(pts)):
        # Fallback to Fibonacci
        pts = _fibonacci_sphere(N)
        pts = _normalize(pts)
        return pts
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Perturb slightly
                pts[j] = pts[j] + 1e-8 * np.random.randn(3)
                pts = _normalize(pts)
                break
    
    return pts
