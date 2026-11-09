"""Optimized logarithmic energy minimization for 150 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and tangent-space gradient for points on the sphere.
    Energy = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)

    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    energy = -math.fsum(math.log(d) for d in d_ij)

    # Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    # grad[i] = -sum_j (x_i - x_j) * inv_dist2[i,j]
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)

    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts

    return energy, grad_tangent


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_initialization(n, rng):
    """Random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _ring_initialization(n, rng):
    """Ring-based initialization for diversity."""
    pts = np.zeros((n, 3))
    num_rings = 8
    pts_per_ring = n // num_rings
    for ring in range(num_rings):
        start = ring * pts_per_ring
        end = min(start + pts_per_ring, n)
        if ring == num_rings - 1:
            end = n
        z = 1.0 - 2.0 * (ring + 0.5) / num_rings
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        for k in range(start, end):
            angle = 2.0 * np.pi * (k - start) / (end - start) + rng.uniform(0, 0.5)
            pts[k] = [r * np.cos(angle), r * np.sin(angle), z]
    return _project_to_sphere(pts)


def _gradient_descent_sphere(pts, max_iter=2000, tol=1e-10, time_limit=None, start_time=None):
    """Gradient descent with adaptive learning rate on the sphere."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = DEADLINE - start_time

    pts = pts.copy()
    energy, grad = _energy_and_grad(pts)
    best_local_energy = energy
    best_local_pts = pts.copy()

    lr = 0.01
    stall_count = 0
    max_stall = 50

    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        new_pts = _project_to_sphere(pts - lr * grad)
        new_energy, new_grad = _energy_and_grad(new_pts)

        if new_energy < energy - 1e-12:
            pts = new_pts
            energy = new_energy
            grad = new_grad
            if energy < best_local_energy:
                best_local_energy = energy
                best_local_pts = pts.copy()
            lr = min(lr * 1.05, 0.05)
            stall_count = 0
        else:
            lr *= 0.7
            stall_count += 1
            if lr < 1e-9 or stall_count > max_stall:
                break

    return best_local_pts, best_local_energy


def _lbfgs_optimize(pts, max_iter=1000, time_limit=None, start_time=None):
    """L-BFGS-B optimization in R^3 with projection."""
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = DEADLINE - start_time

    def objective(x_flat):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise Exception("Time limit exceeded")
        p = x_flat.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': max_iter * 3})
        if res.success or res.status == 0:
            final_pts = _project_to_sphere(res.x.reshape(N, 3))
            final_energy, _ = _energy_and_grad(final_pts)
            return final_pts, final_energy
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    # Initial guess: Fibonacci spiral
    best_pts = _fibonacci_spiral(N)
    best_energy, _ = _energy_and_grad(best_pts)

    # Phase 1: Optimize from Fibonacci spiral with gradient descent
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 20:
        opt_pts, opt_energy = _gradient_descent_sphere(
            best_pts, max_iter=3000, time_limit=min(remaining - 5, 40.0), start_time=start_time)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy

    # Phase 2: L-BFGS-B polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        opt_pts, opt_energy = _lbfgs_optimize(
            best_pts, max_iter=500, time_limit=min(remaining - 5, 30.0), start_time=start_time)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy

    # Phase 3: Multiple restarts with different initializations
    initializations = [
        _random_initialization(N, rng),
        _fibonacci_spiral(N, seed_offset=0.3),
        _fibonacci_spiral(N, seed_offset=0.7),
        _ring_initialization(N, rng),
        _project_to_sphere(_fibonacci_spiral(N) + rng.standard_normal((N, 3)) * 0.1),
        _project_to_sphere(_fibonacci_spiral(N) + rng.standard_normal((N, 3)) * 0.2),
        _random_initialization(N, rng),
        _ring_initialization(N, rng),
    ]

    for init_pts in initializations:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 15:
            break

        # Quick gradient descent to get close
        time_per_gd = min(remaining / 4.0, 10.0)
        opt_pts, opt_energy = _gradient_descent_sphere(
            init_pts, max_iter=1500, time_limit=time_per_gd, start_time=start_time)

        # L-BFGS polish if time permits
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10 and opt_energy < best_energy + 5:
            time_per_lbfgs = min(remaining / 3.0, 15.0)
            opt_pts, opt_energy = _lbfgs_optimize(
                opt_pts, max_iter=300, time_limit=time_per_lbfgs, start_time=start_time)

        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy

    # Final polishing
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        opt_pts, opt_energy = _gradient_descent_sphere(
            best_pts, max_iter=5000, time_limit=min(remaining - 3, 25.0), start_time=start_time)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy

    # Final L-BFGS polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 8:
        opt_pts, opt_energy = _lbfgs_optimize(
            best_pts, max_iter=200, time_limit=min(remaining - 2, 12.0), start_time=start_time)
        if opt_energy < best_energy:
            best_pts = opt_pts
            best_energy = opt_energy

    # Ensure validity
    if np.any(~np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)

    # Check for too-close points
    dist = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        iu = np.triu_indices(N, k=1)
        d_flat = dist[iu]
        min_idx = np.argmin(d_flat)
        i, j = iu[0][min_idx], iu[1][min_idx]
        diff = best_pts[i] - best_pts[j]
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-15:
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(best_pts[i][0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            ortho = ortho - np.dot(ortho, best_pts[i]) * best_pts[i]
            ortho = ortho / np.linalg.norm(ortho)
            diff = ortho
        else:
            diff = diff / norm_diff
        best_pts[i] = _project_to_sphere(best_pts[i] + 1e-8 * diff)
        best_pts[j] = _project_to_sphere(best_pts[j] - 1e-8 * diff)

    return best_pts
