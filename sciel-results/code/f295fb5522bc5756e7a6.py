"""Rewritten program: Stochastic Lloyd's Algorithm and Coordinate Repulsion."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum


def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n):
    """Generate n random points on the unit sphere."""
    v = np.random.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _log_energy(pts):
    """Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            total += -np.log(d)
    return total


def _stochastic_lloyd(pts, n_samples=5000, n_iters=50):
    """
    Stochastic Lloyd's Algorithm for sphere.
    Approximates Voronoi centroids by sampling.
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(n_iters):
        samples = _random_sphere(n_samples)
        dots = samples @ pts.T
        nearest = np.argmax(dots, axis=1)
        
        sum_pts = np.zeros((n, 3))
        counts = np.zeros(n)
        np.add.at(sum_pts, nearest, samples)
        np.add.at(counts, nearest, 1)
        
        for i in range(n):
            if counts[i] > 0:
                new_pt = sum_pts[i] / counts[i]
                norm = np.linalg.norm(new_pt)
                if norm > 1e-12:
                    pts[i] = new_pt / norm
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.where(norms < 1e-12, 1.0, norms)
    return pts


def _coordinate_repulsion(pts, n_iters=500, alpha=0.01):
    """
    Coordinate-wise repulsion (Gauss-Seidel style).
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(n_iters):
        order = np.random.permutation(n)
        for i in order:
            diffs = pts[i] - pts
            dist_sq = np.sum(diffs * diffs, axis=1)
            dist_sq[i] = 1.0 
            inv_dist_sq = 1.0 / dist_sq
            inv_dist_sq[i] = 0.0
            
            force = np.sum(diffs * inv_dist_sq[:, np.newaxis], axis=0)
            dot = np.dot(force, pts[i])
            tangent_force = force - dot * pts[i]
            
            pts[i] = pts[i] + alpha * tangent_force
            norm = np.linalg.norm(pts[i])
            if norm > 1e-12:
                pts[i] = pts[i] / norm
    
    return pts


def _gradient_descent(pts, n_iters=2000, lr_init=0.001, lr_decay=0.999):
    """
    Gradient descent on the logarithmic energy.
    Gradient of log(1/|x_i - x_j|) w.r.t. x_i is:
    (x_i - x_j) / |x_i - x_j|^2
    """
    pts = pts.copy()
    n = len(pts)
    lr = lr_init
    
    for _ in range(n_iters):
        # Compute gradient for each point
        grads = np.zeros_like(pts)
        for i in range(n):
            diffs = pts[i] - pts  # (n, 3)
            dist_sq = np.sum(diffs * diffs, axis=1)  # (n,)
            dist_sq[i] = 1.0
            inv_dist_sq = 1.0 / dist_sq
            inv_dist_sq[i] = 0.0
            # Gradient contribution: sum (x_i - x_j) / |x_i - x_j|^2
            grad_i = np.sum(diffs * inv_dist_sq[:, np.newaxis], axis=0)
            # Project to tangent space
            dot = np.dot(grad_i, pts[i])
            tangent_grad = grad_i - dot * pts[i]
            grads[i] = tangent_grad
        
        # Update
        pts = pts + lr * grads
        
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.where(norms < 1e-12, 1.0, norms)
        
        # Decay learning rate
        lr *= lr_decay
    
    return pts


def construct():
    # Multiple restarts to find better local minima
    np.random.seed(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple different initializations
    for restart in range(5):
        if restart == 0:
            # Start from Fibonacci spiral
            pts = _fibonacci_sphere(N).astype(np.float64)
        else:
            # Random initialization with some structure
            pts = _random_sphere(N)
        
        # Phase 1: Global structure via Stochastic Lloyd's
        pts = _stochastic_lloyd(pts, n_samples=8000, n_iters=25)
        
        # Phase 2: Local refinement via Coordinate Repulsion
        pts = _coordinate_repulsion(pts, n_iters=800, alpha=0.03)
        
        # Phase 3: Fine-tune with gradient descent
        pts = _gradient_descent(pts, n_iters=3000, lr_init=0.002, lr_decay=0.9995)
        
        # Final cleanup
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            continue
        if pts.shape != (N, 3):
            continue
        
        # Check minimum distance
        min_dist = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            continue
        
        # Compute energy
        energy = _log_energy(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # If all restarts failed, fall back to a valid configuration
    if best_pts is None:
        best_pts = _fibonacci_sphere(N).astype(np.float64)
        best_pts = _stochastic_lloyd(best_pts, n_samples=10000, n_iters=30)
        best_pts = _coordinate_repulsion(best_pts, n_iters=1000, alpha=0.05)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    assert np.all(np.isfinite(best_pts))
    assert best_pts.shape == (N, 3)
    
    return best_pts
# EVOLVE-BLOCK-END
