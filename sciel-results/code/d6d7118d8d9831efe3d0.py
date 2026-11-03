"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from math import fsum, log
import time


def _pairwise_distances(points):
    """Compute pairwise chordal distances for points on unit sphere."""
    dots = points @ points.T
    d2 = 2.0 - 2.0 * dots
    np.clip(d2, 0.0, 4.0, out=d2)
    d = np.sqrt(d2)
    return d


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    d = _pairwise_distances(points)
    iu = np.triu_indices(N, k=1)
    dists = d[iu]
    dists = np.maximum(dists, 1e-15)
    vals = -np.log(dists)
    return fsum(vals.tolist())


def _log_energy_grad(points):
    """Compute gradient of log energy w.r.t. points (in R^3).
    Energy = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    d2 = 2.0 - 2.0 * (points @ points.T)
    np.clip(d2, 1e-30, 4.0, out=d2)
    M = 1.0 / d2
    np.fill_diagonal(M, 0.0)
    row_sums = M.sum(axis=1, keepdims=True)
    grad = row_sums * points - M @ points
    return grad


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _project_gradient(points, grad):
    """Project gradient onto tangent space of unit sphere."""
    dots = (points * grad).sum(axis=1, keepdims=True)
    return grad - dots * points


def _fibonacci_spiral(N):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(N, seed=0):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _sobol_points(N, seed=42):
    """Generate quasi-random points on sphere using Halton-like sequence."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 3))
    for i in range(N):
        # Use a simple low-discrepancy approach
        t = (i + 0.5) / N
        z = 1.0 - 2.0 * t
        phi = 2.0 * np.pi * (i * 0.618033988749895)  # golden ratio conjugate
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        pts[i] = [r * np.cos(phi), r * np.sin(phi), z]
    return pts


def _gradient_descent(points, lr, n_iters, momentum=0.9, min_dist=1e-9):
    """Gradient descent on the sphere with momentum."""
    points = points.copy()
    points = _normalize(points)
    v = np.zeros_like(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        v = momentum * v - lr * grad / (grad_norm + 1e-15)
        points = points + v
        points = _normalize(points)
        
        # Check for close points
        d = _pairwise_distances(points)
        if d.min() < min_dist:
            iu = np.triu_indices(N, k=1)
            close_mask = d[iu] < min_dist * 10
            for idx in np.where(close_mask)[0]:
                i, j = iu[0][idx], iu[1][idx]
                mid = (points[i] + points[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-15:
                    mid = mid / mid_norm
                perp = points[i] - points[j]
                perp = perp - np.dot(perp, mid) * mid
                perp_norm = np.linalg.norm(perp)
                if perp_norm < 1e-15:
                    perp = np.array([1.0, 0.0, 0.0])
                    perp = perp - np.dot(perp, mid) * mid
                    perp_norm = np.linalg.norm(perp)
                perp = perp / perp_norm
                points[i] = _normalize(points[i] + 0.01 * perp)
                points[j] = _normalize(points[j] - 0.01 * perp)
        
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    return best_points, best_energy


def _riemannian_gd(points, n_iters=3000, lr_init=0.005, momentum=0.9, T_init=0.5, T_min=0.01, rng=None):
    """Riemannian gradient descent with simulated annealing."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    points = points.copy()
    points = _normalize(points)
    best_energy = _log_energy(points)
    best_points = points.copy()
    v = np.zeros_like(points)
    T = T_init
    
    for it in range(n_iters):
        energy = _log_energy(points)
        grad = _log_energy_grad(points)
        grad = _project_gradient(points, grad)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        lr = lr_init * (0.999 ** it)
        v = momentum * v - lr * grad / (grad_norm + 1e-15)
        new_points = points + v
        new_points = _normalize(new_points)
        
        new_energy = _log_energy(new_points)
        
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        
        T = max(T_min, T_init * (0.999 ** it))
    
    return best_points, best_energy


def _multi_restart(points, n_restarts=5, lr_start=0.01, lr_end=0.0001, n_iters=2000, seed=0):
    """Multi-restart gradient descent."""
    rng = np.random.default_rng(seed)
    best_points = points.copy()
    best_energy = _log_energy(points)
    
    for restart in range(n_restarts):
        if restart == 0:
            pts = points.copy()
        else:
            noise = rng.standard_normal((N, 3)) * (0.05 * (1.0 + restart * 0.5))
            pts = _normalize(points + noise)
        
        lr = lr_start
        for phase in range(3):
            n_phase = n_iters // 3
            lr_phase = lr_start * (lr_end / lr_start) ** (phase / 3.0)
            pts, energy = _gradient_descent(pts, lr_phase, n_phase, momentum=0.9)
            if energy < best_energy:
                best_energy = energy
                best_points = pts.copy()
        
        pts, energy = _gradient_descent(pts, lr_end, n_iters // 5, momentum=0.99)
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    return best_points, best_energy


def construct():
    # Start with multiple initializations
    inits = [
        _fibonacci_spiral(N),
        _sobol_points(N, seed=42),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
    ]
    
    best_points = None
    best_energy = float('inf')
    
    rng = np.random.default_rng(123)
    
    for init_pts in inits:
        init_pts = _normalize(init_pts)
        pts, e = _riemannian_gd(init_pts, n_iters=3000, lr_init=0.005, momentum=0.9, T_init=0.5, T_min=0.01, rng=rng)
        if e < best_energy:
            best_energy = e
            best_points = pts.copy()
            
        if best_points is not None:
            noise = rng.standard_normal((N, 3)) * 0.05
            perturbed = _normalize(best_points + noise)
            pts2, e2 = _riemannian_gd(perturbed, n_iters=1000, lr_init=0.001, momentum=0.9, T_init=0.1, T_min=0.001, rng=rng)
            if e2 < best_energy:
                best_energy = e2
                best_points = pts2.copy()
    
    if best_points is not None:
        best_points, best_energy = _riemannian_gd(best_points, n_iters=1000, lr_init=0.0001, momentum=0.99, T_init=0.01, T_min=0.0001, rng=rng)
    
    best_points = _normalize(best_points)
    d = _pairwise_distances(best_points)
    if d.min() < 1e-9:
        best_points = _fibonacci_spiral(N)
        best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
