"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / di) for di in d)


def _gradient(points):
    """Compute gradient of log energy with respect to points (vectorized)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-30)
    # Gradient: grad[i] = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _optimize_from_start(start_points, max_iter=5000, tol=1e-12):
    """Optimize points on sphere using projected gradient with adaptive step size."""
    points = start_points.copy()
    energy = _log_energy(points)
    
    # Adaptive learning rate
    lr = 0.05
    best_energy = energy
    best_points = points.copy()
    no_improve_count = 0
    
    for it in range(max_iter):
        grad = _gradient(points)
        # Project gradient to tangent space
        grad_t = grad - np.sum(grad * points, axis=1, keepdims=True) * points
        
        # Adaptive step size: try the step, if it doesn't improve, halve lr
        accepted = False
        for _ in range(20):
            new_points = points - lr * grad_t
            new_points = _project_to_sphere(new_points)
            new_energy = _log_energy(new_points)
            
            if new_energy < energy - tol:
                points = new_points
                energy = new_energy
                accepted = True
                if energy < best_energy:
                    best_energy = energy
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                break
            else:
                lr *= 0.5
        
        if not accepted:
            lr = 0.05
            no_improve_count += 1
        
        if no_improve_count > 100:
            break
    
    return best_points, best_energy


def construct():
    # Generate multiple diverse starting points
    rng = np.random.default_rng(42)
    
    candidates = []
    energies = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    fib_points = _project_to_sphere(fib_points)
    candidates.append(fib_points)
    
    # 2. Random points
    for _ in range(5):
        pts = rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        candidates.append(pts)
    
    # 3. Points with some structure: icosahedral-like starting points
    # Use a perturbed regular polyhedron
    for scale in [0.1, 0.3, 0.5]:
        base = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1, -1], [0, -1, 1], [1, 0, -1], [-1, 0, 1],
            [0, 1, 1], [0, -1, -1], [1, 0, 1], [-1, 0, -1]
        ], dtype=float)
        # Perturb and tile to get 100 points
        n_base = len(base)
        idx = (np.arange(N) * 7) % n_base  # Use a non-trivial mapping
        pts = base[idx] + scale * rng.standard_normal((N, 3))
        pts = _project_to_sphere(pts)
        candidates.append(pts)
    
    # Optimize each candidate
    best_points = None
    best_energy = np.inf
    
    for cand in candidates:
        pts, energy = _optimize_from_start(cand, max_iter=3000)
        if energy < best_energy:
            best_energy = energy
            best_points = pts
    
    return best_points
# EVOLVE-BLOCK-END
