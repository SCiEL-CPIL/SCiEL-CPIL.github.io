"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

DEADLINE = 115.0  # safety margin before 120s limit


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    # Compute pairwise distances using broadcasting
    # To save memory for large N, we can do it in chunks, but N=100 is small
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal and upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    # Avoid log(0) or log of very small numbers
    # If any distance is < 1e-9, the energy is effectively -inf (or very negative)
    # But the task says points closer than 1e-9 make it invalid.
    # We assume valid inputs during optimization.
    return math.fsum(math.log(1.0 / di) for di in d)


def _energy_fast(pts):
    """Vectorized energy calculation for speed during optimization."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    # Clip to avoid log(0)
    d = np.clip(d, 1e-15, None)
    return np.sum(np.log(1.0 / d))


def _gradient(pts):
    """Compute gradient of the energy with respect to points on the sphere."""
    # This is complex to derive correctly with constraints.
    # Instead, we'll use finite differences or a trust-region method that handles constraints.
    # However, for a 'stronger algorithm', let's implement a proper projected gradient step
    # or use L-BFGS-B with bounds and then project.
    # Actually, the most robust way in SciPy is to use `minimize` with `method='SLSQP'`
    # or `trust-constr`, but they can be slow.
    # Let's stick to a robust local optimization strategy:
    # 1. Start from a good initial guess (Fibonacci spiral).
    # 2. Use multiple restarts with small perturbations.
    # 3. Use `minimize` with `method='L-BFGS-B'` on the sphere coordinates?
    #    Better: use spherical coordinates or quaternion?
    #    Simplest robust approach: optimize in R^3 with equality constraint |x|^2=1.
    #    SciPy's `minimize` with `constraints` can handle this.

    # For speed and reliability, let's use a simple iterative repulsion method (Lennard-Jones-like)
    # or a gradient descent with projection onto the sphere.
    # The gradient of E = sum log(1/d_ij) = -sum log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Then project the gradient onto the tangent space of the sphere.

    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-15:
                continue
            grad[i] += d / dist_sq
    # The energy is sum log(1/d) = -sum log(d)
    # d/dx_i (-log d_ij) = - (1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # Wait, the force is attractive or repulsive?
    # Energy = sum log(1/d) = - sum log(d).
    # If d decreases, log(1/d) increases (energy goes up). So we want to maximize d?
    # No, the task says minimize the logarithmic energy.
    # log(1/d) = -log(d). To minimize -log(d), we want to maximize d.
    # So the force should push points apart.
    # Gradient of -log(d) w.r.t x_i is -(x_i - x_j)/d^2.
    # So grad[i] should be sum_{j!=i} -(x_i - x_j)/d^2 = sum_{j!=i} (x_j - x_i)/d^2.
    # My loop above computed sum (x_i - x_j)/d^2, which is the negative of the gradient.
    # So the gradient is -grad.
    # To minimize, we move in the direction of -gradient.
    # So we move in the direction of grad (as computed above).
    # Let's re-verify:
    # E = -log(d). dE/dx_i = -1/d * dd/dx_i. dd/dx_i = (x_i - x_j)/d.
    # So dE/dx_i = -(x_i - x_j)/d^2.
    # My code: grad[i] += (x_i - x_j)/d^2.
    # So grad is the negative of the true gradient.
    # Descent direction is -gradient = grad.
    # So we update x_i <- x_i + step * grad[i].
    # Then project back to sphere.

    return -grad  # Return the true gradient


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def _local_optimize(pts, max_iter=1000, step_size=0.01, tol=1e-10):
    """Gradient descent with projection onto the sphere."""
    current_energy = _energy_fast(pts)
    for _ in range(max_iter):
        grad = _gradient(pts)
        # Normalize gradient to have consistent step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        grad = grad / grad_norm
        # Adaptive step size
        new_pts = _project_to_sphere(pts + step_size * grad)
        new_energy = _energy_fast(new_pts)
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            # Decrease step size slightly
            step_size *= 0.99
        else:
            # Backtracking
            step_size *= 0.5
            if step_size < 1e-8:
                break
    return pts, current_energy


def construct():
    start_time = time.time()

    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_pts = pts.copy()
    best_energy = _energy_fast(pts)

    # Try a few different random restarts and local optimizations
    rng = np.random.default_rng(42)

    # First, optimize the initial guess
    optimized, energy = _local_optimize(pts.copy(), max_iter=2000, step_size=0.05)
    if energy < best_energy:
        best_energy = energy
        best_pts = optimized

    # Check time
    if time.time() - start_time > DEADLINE - 10:
        return _project_to_sphere(best_pts)

    # Try random restarts
    for restart in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Generate random points on the sphere
        random_pts = rng.standard_normal((N, 3))
        random_pts = _project_to_sphere(random_pts)

        # Optimize
        optimized, energy = _local_optimize(random_pts, max_iter=1000, step_size=0.05)
        if energy < best_energy:
            best_energy = energy
            best_pts = optimized

    # Final polish with smaller step size
    if time.time() - start_time < DEADLINE - 2:
        polished, energy = _local_optimize(best_pts.copy(), max_iter=5000, step_size=0.01, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_pts = polished

    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
