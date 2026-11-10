import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 590  # Safety margin

    def score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def objective(x):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1.0
        d_min = np.min(dists)
        return -(d_min / d_max) ** 2

    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    best_points = None
    best_score = 0.0

    # Initial candidate: Fibonacci sphere
    init_pts = fibonacci_sphere(n)
    best_points = init_pts.copy()
    best_score = score(init_pts)

    # Soft-min / log-sum-exp smoothed objective for global search
    # This makes the objective smooth and amenable to gradient-based methods
    def soft_objective(x, beta=50.0):
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists_sq = pdist(p) ** 2
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 1.0
        # Soft-min of distances, scaled by d_max
        # We want to maximize (d_min/d_max)^2 = d_min^2 / d_max^2
        # Minimize -log(sum(exp(-beta * dists_sq/d_max_sq))) / beta
        normalized = dists_sq / d_max_sq
        # Avoid overflow
        min_norm = np.min(normalized)
        log_sum = -beta * min_norm + np.log(np.sum(np.exp(-beta * (normalized - min_norm))))
        soft_min_norm = log_sum / beta
        return -soft_min_norm + 0.01 * (1.0 - soft_min_norm)

    rng = np.random.default_rng(42)

    # Phase 1: Global search with soft objective and multiple starts
    # Use L-BFGS-B which is faster than Powell for smooth objectives
    soft_restarts = 0
    max_soft_restarts = 30

    while time.time() - start_time < time_limit - 60 and soft_restarts < max_soft_restarts:
        if soft_restarts == 0:
            x0 = best_points.ravel()
        elif soft_restarts == 1:
            # Try icosahedron-based initialization
            # 12 vertices of icosahedron + 2 poles
            phi_g = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [0, 1, phi_g], [0, -1, phi_g], [0, 1, -phi_g], [0, -1, -phi_g],
                [1, phi_g, 0], [-1, phi_g, 0], [1, -phi_g, 0], [-1, -phi_g, 0],
                [phi_g, 0, 1], [phi_g, 0, -1], [-phi_g, 0, 1], [-phi_g, 0, -1]
            ])
            norms = np.linalg.norm(ico, axis=1, keepdims=True)
            ico = ico / norms
            # Add 2 more points at poles
            extra = np.array([[0, 1, 0], [0, -1, 0]])
            x0 = np.vstack([ico, extra]).ravel()
        else:
            # Random restarts with perturbation
            x0 = best_points.ravel() + rng.normal(0, 0.3, size=(n, d)).ravel()

        try:
            res = minimize(
                soft_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            soft_restarts += 1

            candidate = res.x.reshape(n, d)
            cand_score = score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate

        except Exception:
            pass

    # Phase 2: Local refinement with exact objective using Powell
    # This polishes the best candidate found
    local_restarts = 0
    max_local_restarts = 50

    while time.time() - start_time < time_limit - 10 and local_restarts < max_local_restarts:
        if local_restarts == 0:
            x0 = best_points.ravel()
        else:
            # Small perturbations around best point
            x0 = best_points.ravel() + rng.normal(0, 0.05, size=(n, d)).ravel()

        try:
            res = minimize(
                objective,
                x0,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            local_restarts += 1

            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)

        except Exception:
            pass

    # Phase 3: Final high-precision polish with Nelder-Mead
    if time.time() - start_time < time_limit - 5:
        try:
            res = minimize(
                objective,
                best_points.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
        except Exception:
            pass

    # Final check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)

    return best_points
