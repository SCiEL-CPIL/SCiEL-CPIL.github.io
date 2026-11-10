"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute log-energy and its gradient for points on the unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  for each i
    Uses squared distances to avoid sqrt where possible.
    """
    n = pts.shape[0]
    # Squared pairwise distances: |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2 x_i.x_j
    # For unit sphere points, |x_i|^2 = 1, so dist2 = 2 - 2*x_i.x_j
    dot = pts @ pts.T  # (n, n)
    dist2 = 2.0 - 2.0 * dot
    np.maximum(dist2, 1e-18, out=dist2)  # clamp to avoid log(0) or div by zero
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij) = -0.5 * sum_{i<j} log(dist2_ij)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist2[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (x_j - x_i) / dist_ij^2
    # = sum_{j != i} (x_j - x_i) / dist2_ij
    inv_dist2 = np.zeros_like(dist2)
    inv_dist2[iu] = 1.0 / dist2[iu]
    inv_dist2 = (inv_dist2 + inv_dist2.T)  # symmetric
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j inv_dist2[i,j] * (pts[j] - pts[i])
    # = sum_j inv_dist2[i,j] * pts[j] - pts[i] * sum_j inv_dist2[i,j]
    grad = (inv_dist2 @ pts) - pts * inv_dist2.sum(axis=1, keepdims=True)
    
    return energy, grad


def _fibonacci_sphere(n, seed=0):
    """Generate n points on unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed * 0.1
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _random_sphere(n, rng):
    """Generate n random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_and_normalize(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _optimize_from_start(pts0):
    """Run L-BFGS-B from an initial configuration, returning best energy and points."""
    pts = pts0.copy()
    
    def objective(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        p = _project_and_normalize(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space: g_tan = g - (g·p_hat)p_hat
        g_tan = g - (g * p).sum(axis=1, keepdims=True) * p
        return g_tan.ravel()
    
    x0 = pts0.ravel()
    result = minimize(
        objective, x0, jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    pts_opt = _project_and_normalize(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(pts_opt)
    return energy, pts_opt


def _polish(pts, max_iter=2000, tol=1e-14):
    """High-precision coordinate-descent polish on the unit sphere.
    For each point, move it along the direction that most reduces energy,
    using a small step size and re-normalizing.
    """
    n = pts.shape[0]
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            old_energy = energy
            # Try small perturbations in each coordinate direction
            # Use gradient-based step: move against projected gradient
            _, grad = _energy_and_grad(pts)
            g_i = grad[i]
            # Project to tangent space
            g_tan = g_i - np.dot(g_i, pts[i]) * pts[i]
            g_norm = np.linalg.norm(g_tan)
            if g_norm < 1e-15:
                continue
            # Try a few step sizes
            for step_factor in [0.1, 0.05, 0.01, 0.005]:
                step = step_factor * g_tan / g_norm
                new_pt = pts[i] + step
                new_pt = new_pt / np.linalg.norm(new_pt)
                pts[i] = new_pt
                new_energy, _ = _energy_and_grad(pts)
                if new_energy < old_energy - 1e-16:
                    energy = new_energy
                    improved = True
                    break
                else:
                    pts[i] = new_pt  # keep trying smaller steps
            # If no improvement, revert
            if not improved and pts[i] is not None:
                # Already reverted in loop above if no improvement
                pass
        if not improved:
            break
    
    return energy, pts


def construct():
    """Multi-start optimization of logarithmic energy on the unit sphere."""
    t_start = time.time()
    best_energy = float('inf')
    best_pts = None
    
    starts = []
    
    # Start 1: Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # Start 2: Random with seed 42
    rng = np.random.default_rng(42)
    starts.append(_random_sphere(N, rng))
    
    # Start 3: Fibonacci with offset
    starts.append(_fibonacci_sphere(N, seed=1.0))
    
    # Start 4: Another random
    rng = np.random.default_rng(123)
    starts.append(_random_sphere(N, rng))
    
    # Start 5: Fibonacci with different seed
    starts.append(_fibonacci_sphere(N, seed=2.5))
    
    # Start 6: Random with another seed
    rng = np.random.default_rng(999)
    starts.append(_random_sphere(N, rng))
    
    for idx, pts0 in enumerate(starts):
        if time.time() > _DEADLINE - 15:
            break
        energy, pts_opt = _optimize_from_start(pts0)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
    
    # Final high-precision polish pass
    if best_pts is not None and time.time() < _DEADLINE - 3:
        energy, pts_polished = _polish(best_pts, max_iter=1000, tol=1e-15)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_polished
    
    return best_pts
# EVOLVE-BLOCK-END
