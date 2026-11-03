"""Improved program for sphere covering with 50 points."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees using the convex hull approach."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0

    max_circumradius = 0.0
    facets = hull.simplices

    for simplex in facets:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-15:
            continue
        normal = normal / norm

        for p in [p1, p2, p3]:
            cos_angle = np.dot(normal, p)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > max_circumradius:
                max_circumradius = angle

    return np.degrees(max_circumradius)


def _riesz_energy(pts, s=12.0):
    """Compute the Riesz s-energy sum over all pairs."""
    pts = _normalize(pts)
    n = len(pts)
    # Compute all pairwise dot products
    dots = pts @ pts.T
    np.fill_diagonal(dots, 2.0)  # Avoid self-interaction
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distances
    angles = np.arccos(dots)
    # Riesz potential: sum of 1/d^s for all pairs
    # Use log-sum-exp for numerical stability
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    # Only compute for upper triangle to avoid double counting
    triu_indices = np.triu_indices(n, k=1)
    pair_angles = angles[triu_indices]
    # Riesz energy: sum of (1/cos(theta))^s = sec^s(theta)
    # Or equivalently sum of (1/sin(theta))^s for angular distance
    # Use: sum of (1 - cos(theta))^-s/2 for stability
    cos_vals = dots[triu_indices]
    cos_vals = np.clip(cos_vals, -1.0 + 1e-15, 1.0 - 1e-15)
    # Riesz s-energy = sum (1/d^s) where d is chordal distance
    # chordal distance squared = 2*(1-cos(theta))
    d2 = 2.0 * (1.0 - cos_vals)
    d2 = np.maximum(d2, 1e-15)
    energy = np.sum(d2 ** (-s / 2.0))
    return energy


def _riesz_energy_and_grad(pts, s=12.0):
    """Compute Riesz s-energy and its gradient w.r.t. points on sphere."""
    pts = _normalize(pts)
    n = len(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 2.0)
    dots = np.clip(dots, -1.0, 1.0)

    # Riesz energy using chordal distance
    d2 = 2.0 * (1.0 - dots)
    np.fill_diagonal(d2, 1.0)  # Avoid self-interaction in gradient
    d2 = np.maximum(d2, 1e-15)

    # Energy = sum_{i<j} d2[i,j]^(-s/2)
    half_s = s / 2.0
    triu_i, triu_j = np.triu_indices(n, k=1)
    d2_pairs = d2[triu_i, triu_j]
    energy = np.sum(d2_pairs ** (-half_s))

    # Gradient: dE/dpts[k] = -s * sum_{j!=k} d2[k,j]^(-s/2 - 1) * (pts[k] - pts[j])
    # But we need to project onto tangent space of sphere
    grad = np.zeros_like(pts)
    for k in range(n):
        for j in range(n):
            if k == j:
                continue
            d2_kj = d2[k, j]
            d2_kj = max(d2_kj, 1e-15)
            # dE/dpts[k] contribution from pair (k,j)
            # E includes d2^(-s/2) = (2(1-dots))^(-s/2)
            # d(d2)/dpts[k] = -2*(pts[k]-pts[j])
            # dE/d(d2) = -s/2 * d2^(-s/2 - 1)
            # dE/dpts[k] = dE/d(d2) * d(d2)/dpts[k] = s * d2^(-s/2-1) * (pts[k]-pts[j])
            coeff = s * d2_kj ** (-half_s - 1.0)
            grad[k] += coeff * (pts[k] - pts[j])

    # Project gradient onto tangent space of sphere
    # Tangent projection: grad_tangent = grad - (grad·pts)pts
    for k in range(n):
        dot_val = np.dot(grad[k], pts[k])
        grad[k] -= dot_val * pts[k]

    return energy, grad


def _riesz_energy_grad_fast(pts, s=12.0):
    """Fast vectorized computation of Riesz s-energy and gradient."""
    pts = _normalize(pts)
    n = len(pts)
    dots = pts @ pts.T
    np.fill_diagonal(dots, 2.0)
    dots = np.clip(dots, -1.0, 1.0)

    d2 = 2.0 * (1.0 - dots)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)

    half_s = s / 2.0
    # Energy
    triu_i, triu_j = np.triu_indices(n, k=1)
    d2_pairs = d2[triu_i, triu_j]
    energy = np.sum(d2_pairs ** (-half_s))

    # Gradient computation (vectorized)
    # grad[k] = s * sum_{j!=k} d2[k,j]^(-s/2-1) * (pts[k] - pts[j])
    d2_grad = d2 ** (-half_s - 1.0)
    np.fill_diagonal(d2_grad, 0.0)  # No self-interaction

    # grad[k] = s * (sum_j d2_grad[k,j] * pts[k] - sum_j d2_grad[k,j] * pts[j])
    # = s * (pts[k] * sum_j d2_grad[k,j] - (d2_grad @ pts)[k])
    row_sums = np.sum(d2_grad, axis=1)
    grad = s * (pts * row_sums[:, np.newaxis] - d2_grad @ pts)

    # Project onto tangent space
    dot_vals = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot_vals * pts

    return energy, grad


def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # Start with Fibonacci spiral
    points = _fibonacci_points()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)

    # Optimize using Riesz s-energy with L-BFGS-B
    # We optimize in Cartesian coordinates but project back to sphere
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        energy, grad = _riesz_energy_grad_fast(pts, s=12.0)
        return energy, grad.flatten()

    # L-BFGS-B optimization
    try:
        result = minimize(
            objective_and_grad,
            points.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < 1e6:
            opt_pts = _normalize(result.x.reshape(N, 3))
            opt_radius = _covering_radius(opt_pts)
            if opt_radius < best_radius:
                best_points = opt_pts
                best_radius = opt_radius
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_points

    # Try multiple restarts with different s values and initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break

        # Different initializations
        if restart == 0:
            init_pts = _fibonacci_points()
        elif restart == 1:
            init_pts = best_points + np.random.randn(N, 3) * 0.3
            init_pts = _normalize(init_pts)
        elif restart == 2:
            # Random initialization
            init_pts = np.random.randn(N, 3)
            init_pts = _normalize(init_pts)
        elif restart == 3:
            # Jittered Fibonacci
            jitter = np.random.randn(N, 3) * 0.1
            init_pts = _fibonacci_points() + jitter
            init_pts = _normalize(init_pts)
        else:
            init_pts = best_points + np.random.randn(N, 3) * 0.15
            init_pts = _normalize(init_pts)

        # Try different s values
        s_val = [8.0, 12.0, 16.0, 10.0, 14.0][restart % 5]

        def obj_grad(x, s=s_val):
            pts = x.reshape(N, 3)
            energy, grad = _riesz_energy_grad_fast(pts, s=s)
            return energy, grad.flatten()

        try:
            result = minimize(
                obj_grad,
                init_pts.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.success or result.fun < 1e6:
                opt_pts = _normalize(result.x.reshape(N, 3))
                opt_radius = _covering_radius(opt_pts)
                if opt_radius < best_radius:
                    best_points = opt_pts
                    best_radius = opt_radius
        except Exception:
            pass

    # Final local polish using gradient descent on covering radius proxy
    if time.time() < deadline - 20.0:
        # Do a few steps of gradient descent minimizing the max pairwise angular distance
        # as a proxy for covering radius
        current = best_points.copy()
        for step in range(100):
            if time.time() > deadline:
                break
            # Compute angular distances
            dots = current @ current.T
            np.fill_diagonal(dots, 2.0)
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            # Max distance from each point to its nearest neighbor
            # Use soft max for smoothness
            temp = 0.05
            log_max = np.logsumexp(angles / temp, axis=1) * temp
            # Gradient of soft max
            weights = np.exp((angles - log_max[:, np.newaxis]) / temp)
            weights /= weights.sum(axis=1, keepdims=True)
            # d(angles)/d(dots) = -1/sqrt(1-dots^2)
            d_angles_ddots = -1.0 / np.sqrt(np.maximum(1.0 - dots ** 2, 1e-10))
            np.fill_diagonal(d_angles_ddots, 0.0)
            # Gradient w.r.t. points
            grad_dots = weights @ d_angles_ddots
            grad_pts = 2.0 * (current * grad_dots.sum(axis=1, keepdims=True) - grad_dots @ current)
            # Project to tangent space
            dot_vals = np.sum(grad_pts * current, axis=1, keepdims=True)
            grad_pts -= dot_vals * current
            # Step
            lr = 0.01 / (1.0 + step * 0.1)
            current = current - lr * grad_pts
            current = _normalize(current)
            cur_radius = _covering_radius(current)
            if cur_radius < best_radius:
                best_points = current
                best_radius = cur_radius

    return best_points
# EVOLVE-BLOCK-END
