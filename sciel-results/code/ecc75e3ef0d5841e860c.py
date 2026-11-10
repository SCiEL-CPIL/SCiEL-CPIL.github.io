"""Rewritten program for packing 50 equal circles in a unit circle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50


def construct():
    """
    Find the centers of N=50 equal circles inside the unit circle
    maximizing the minimum distance between centers and from boundary.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin before 160s kill (but spec says 120s eval, 150s wall)
    # Actually spec says 120s evaluation, 150s wall-clock kill. Let's use 145s to be safe.
    deadline = start_time + 145.0

    best_score = 0.0
    best_centers = None

    def compute_score(centers):
        """
        Compute the score: min of (min pairwise distance / 2, min (1 - |center|))
        Centers must be inside the unit circle (|c| <= 1) and distinct.
        """
        # Check validity
        if not np.all(np.isfinite(centers)):
            return -np.inf
        if centers.shape != (N, 2):
            return -np.inf

        # Distance from origin
        dists_origin = np.sqrt(np.sum(centers**2, axis=1))
        # Boundary constraint: 1 - |c_i|
        boundary_margins = 1.0 - dists_origin
        if np.any(boundary_margins < -1e-10):
            return -np.inf

        # Pairwise distances
        # Use broadcasting for efficiency
        # d[i,j] = |c_i - c_j|
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Get minimum pairwise distance (excluding self)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)

        score = min(min_pairwise / 2.0, np.min(boundary_margins))
        return score

    def initial_guess(ring_structure):
        """
        Generate an initial guess based on a ring structure.
        ring_structure: list of (count, radius) tuples.
        """
        centers = np.zeros((N, 2))
        idx = 0
        for count, radius in ring_structure:
            if idx >= N:
                break
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for angle in angles:
                if idx >= N:
                    break
                centers[idx, 0] = radius * np.cos(angle)
                centers[idx, 1] = radius * np.sin(angle)
                idx += 1
        # If we didn't fill all, add some random points near center
        if idx < N:
            remaining = N - idx
            angles = np.linspace(0, 2 * np.pi, remaining, endpoint=False)
            for i, angle in enumerate(angles):
                centers[idx + i, 0] = 0.2 * np.cos(angle)
                centers[idx + i, 1] = 0.2 * np.sin(angle)
        return centers

    # Try several initial structures
    # Structure 1: 1 center + 6 ring + 12 ring + 15 ring + 16 ring?
    # Total 50.
    # Let's try a few plausible ring configurations.
    # Known packing has some structure.
    # Let's try:
    # 1. 1 center, 6 at r~0.2, 12 at r~0.4, 15 at r~0.6, 16 at r~0.8
    # 2. 1 center, 6 at r~0.15, 12 at r~0.35, 15 at r~0.55, 16 at r~0.75
    # 3. Random initialization

    structures = [
        [(1, 0.0), (6, 0.2), (12, 0.4), (15, 0.6), (16, 0.8)],
        [(1, 0.0), (6, 0.15), (12, 0.35), (15, 0.55), (16, 0.75)],
        [(1, 0.0), (6, 0.25), (12, 0.5), (15, 0.7), (16, 0.9)],
        [(1, 0.0), (6, 0.1), (12, 0.3), (15, 0.5), (16, 0.7)],
    ]

    # Add random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random points inside circle
        r = np.sqrt(rng.random(N)) * 0.95
        theta = rng.random(N) * 2 * np.pi
        centers = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        structures.append(centers)

    for init in structures:
        if time.time() > deadline:
            break

        if isinstance(init, list):
            centers = initial_guess(init)
        else:
            centers = init.copy()

        # Optimize
        # We want to maximize the minimum distance.
        # Equivalent to minimizing the negative of the minimum distance.
        # But min is non-differentiable.
        # Use a smooth approximation or just use a local solver with finite differences?
        # scipy.optimize.minimize with method='L-BFGS-B' or 'BFGS' requires gradient.
        # We can compute gradient numerically or analytically.
        # Let's use a simple approach: minimize the sum of squared violations of a target radius.
        # Or better: maximize r such that all constraints are satisfied.
        # This is a non-linear programming problem.
        # Let's use scipy.optimize.minimize on a function that penalizes violations.

        def objective(x):
            c = x.reshape(N, 2)
            # Compute min pairwise distance
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pairwise = np.min(dists)

            # Compute min boundary margin
            dists_origin = np.sqrt(np.sum(c**2, axis=1))
            min_boundary = np.min(1.0 - dists_origin)

            # We want to maximize min(min_pairwise/2, min_boundary)
            # Let's minimize -min(min_pairwise/2, min_boundary)
            # But this is non-smooth.
            # Instead, let's minimize a function that encourages all distances to be large.
            # A common trick: minimize the sum of (target - dist)^2 for dist < target.
            # But we don't know target.
            # Alternative: use a log-sum-exp based soft-min.
            # soft_min(dists, boundary_margins)
            # Let's try to maximize the minimum by minimizing the maximum violation.

            # Let's use a simpler heuristic: minimize the sum of squared distances to a target radius r.
            # But r is not fixed.

            # Let's use the fact that the optimal solution has many constraints active.
            # We can use a local search approach.
            # For now, let's use scipy.optimize.minimize with a custom objective that is differentiable.
            # Objective: -soft_min(pairwise_dists, boundary_margins)
            # soft_min(x, t) = -log(sum(exp(-x_i/t))) * t
            # This is smooth for t > 0.

            t = 0.05  # Temperature for soft-min
            pairwise_vals = dists / 2.0
            boundary_vals = 1.0 - dists_origin

            # Combine all constraints into one array
            all_vals = np.concatenate([pairwise_vals[np.triu_indices(N, 1)], boundary_vals])
            # Soft min
            log_sum_exp = np.log(np.sum(np.exp(-all_vals / t)))
            soft_min_val = -t * log_sum_exp

            return -soft_min_val

        # Bounds: centers must be inside the unit circle
        bounds = [(-1.0, 1.0)] * (2 * N)

        # Use L-BFGS-B
        try:
            result = minimize(objective, centers.flatten(), method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            optimized_centers = result.x.reshape(N, 2)
            score = compute_score(optimized_centers)
            if score > best_score:
                best_score = score
                best_centers = optimized_centers.copy()
        except Exception as e:
            pass

    # Final polish: try to improve with a local search
    if best_centers is not None:
        # Do a few iterations of local optimization
        for _ in range(3):
            if time.time() > deadline:
                break
            centers = best_centers.copy()

            def objective2(x):
                c = x.reshape(N, 2)
                diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_pairwise = np.min(dists)
                dists_origin = np.sqrt(np.sum(c**2, axis=1))
                min_boundary = np.min(1.0 - dists_origin)
                t = 0.02
                pairwise_vals = dists / 2.0
                boundary_vals = 1.0 - dists_origin
                all_vals = np.concatenate([pairwise_vals[np.triu_indices(N, 1)], boundary_vals])
                log_sum_exp = np.log(np.sum(np.exp(-all_vals / t)))
                soft_min_val = -t * log_sum_exp
                return -soft_min_val

            bounds = [(-1.0, 1.0)] * (2 * N)
            try:
                result = minimize(objective2, centers.flatten(), method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
                optimized_centers = result.x.reshape(N, 2)
                score = compute_score(optimized_centers)
                if score > best_score:
                    best_score = score
                    best_centers = optimized_centers.copy()
            except Exception as e:
                pass

    if best_centers is None:
        # Fallback: return the initial hexagonal grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            best_centers = points[np.argsort(radius, kind="stable")[:N]]
        else:
            best_centers = np.zeros((N, 2))

    return best_centers
