"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0  # safety margin before 120s kill


def _energy_and_grad(pts):
    """Compute log-energy and its gradient for points on the unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2  for each i
    Uses dot products for speed.
    """
    n = pts.shape[0]
    # Squared distances via dot product: |x_i - x_j|^2 = 2 - 2*x_i.x_j for unit vectors
    dot = pts @ pts.T
    dist2 = 2.0 - 2.0 * dot
    np.maximum(dist2, 1e-30, out=dist2)
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: sum over i<j of -0.5*log(dist2_ij)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist2[iu]))
    
    # Gradient: grad_i = sum_{j != i} (x_j - x_i) / dist2_ij
    inv_dist2 = np.zeros_like(dist2)
    inv_dist2[iu] = 1.0 / dist2[iu]
    inv_dist2 = inv_dist2 + inv_dist2.T
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j inv_dist2[i,j] * pts[j] - pts[i] * sum_j inv_dist2[i,j]
    row_sums = inv_dist2.sum(axis=1, keepdims=True)
    grad = (inv_dist2 @ pts) - pts * row_sums
    
    return energy, grad


def _fibonacci_sphere(n, seed=0):
    """Generate n points on unit sphere using golden angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + seed * 0.1
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _halton_sphere(n, base=2):
    """Generate n points on unit sphere using Halton sequence for 3D low-discrepancy."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        # Use different bases for 3 dimensions
        u1 = halton(i + 1, 2)
        u2 = halton(i + 1, 3)
        u3 = halton(i + 1, 5)
        # Map to sphere using spherical coordinates
        theta = 2.0 * np.pi * u1
        z = 1.0 - 2.0 * u2
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
        pts[i, 2] = z
    
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


def _optimize_from_start(pts0, maxiter=500, ftol=1e-14, gtol=1e-10):
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
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    pts_opt = _project_and_normalize(result.x.reshape(-1, 3))
    energy, _ = _energy_and_grad(pts_opt)
    return energy, pts_opt


def _polish(pts, maxiter=1000, ftol=1e-16, gtol=1e-12):
    """High-precision polish pass with very tight tolerances."""
    return _optimize_from_start(pts, maxiter=maxiter, ftol=ftol, gtol=gtol)


def construct():
    """Multi-start optimization of logarithmic energy on the unit sphere."""
    t_start = time.time()
    best_energy = float('inf')
    best_pts = None
    
    # Start 1: Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    # Check time
    if time.time() > _DEADLINE - 40:
        # Quick polish
        energy, pts_opt = _polish(best_pts, maxiter=500, ftol=1e-15, gtol=1e-11)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_opt
        return best_pts
    
    # Start 2: Halton sequence
    pts0 = _halton_sphere(N)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 30:
        return best_pts
    
    # Start 3: Random with seed 42
    rng = np.random.default_rng(42)
    pts0 = _random_sphere(N, rng)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 20:
        return best_pts
    
    # Start 4: Fibonacci with offset
    pts0 = _fibonacci_sphere(N, seed=1.0)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 15:
        return best_pts
    
    # Start 5: Another random
    rng = np.random.default_rng(123)
    pts0 = _random_sphere(N, rng)
    energy, pts_opt = _optimize_from_start(pts0)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    if time.time() > _DEADLINE - 10:
        return best_pts
    
    # Final high-precision polish on the best configuration
    energy, pts_opt = _polish(best_pts, maxiter=1000, ftol=1e-16, gtol=1e-12)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts_opt
    
    return best_pts
# EVOLVE-BLOCK-END
