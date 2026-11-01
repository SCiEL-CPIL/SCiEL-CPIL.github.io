import numpy as np
from scipy.optimize import minimize
import time
import random

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, safety margin

    best_result = None
    best_score = -np.inf

    # Soft-min/soft-max for smooth constraints
    def soft_min(a, b, k=200.0):
        """Smooth minimum: a + b - soft_max(a, b)"""
        m = np.maximum(a, b)
        return m - (1.0 / k) * np.logaddexp(k * (a - m), k * (b - m))

    def penalty_function(params):
        n = N
        xy = params[:2 * n].reshape(n, 2)
        r = params[2 * n:]

        # Boundary constraints (smooth)
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        # Smooth penalty: sum of squared negative distances
        boundary_viol = np.sum(np.maximum(0, -d_left) ** 2) + \
                        np.sum(np.maximum(0, -d_right) ** 2) + \
                        np.sum(np.maximum(0, -d_bottom) ** 2) + \
                        np.sum(np.maximum(0, -d_top) ** 2)

        # Overlap constraints (smooth)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)

        # Use smooth max for overlap
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap ** 2)

        # Also add a small penalty for very small radii to encourage growth
        small_r_penalty = np.sum(np.maximum(0, 0.01 - r) ** 2)

        return -np.sum(r) + 200 * boundary_viol + 200 * overlap_viol + 10 * small_r_penalty

    def is_valid(result, tol=1e-10):
        n = N
        xy = result[:2 * n].reshape(n, 2)
        r = result[2 * n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist_sq = dx ** 2 + dy ** 2
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist_sq[iu] < (min_dist[iu] - tol) ** 2):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2 * n:]
        return np.sum(r)

    def push_apart_vectorized(params, n_iters=50, rng=None):
        """Vectorized push-apart for speed."""
        n = N
        xy = params[:2 * n].reshape(n, 2).copy()
        r = params[2 * n:].copy()
        if rng is None:
            rng = np.random

        for _ in range(n_iters):
            # Compute all pair distances
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist_sq = dx ** 2 + dy ** 2
            dist = np.sqrt(dist_sq)
            min_dist = r[:, None] + r[None, :]

            # Force matrix: positive when overlapping
            iu = np.triu_indices(n, k=1)
            overlaps = np.maximum(0, min_dist[iu] - dist[iu])

            if np.max(overlaps) < 1e-12:
                # Only boundary check
                boundary_ok = True
                if np.any(xy[:, 0] < r - 1e-12) or np.any(xy[:, 0] > 1 - r + 1e-12):
                    boundary_ok = False
                if np.any(xy[:, 1] < r - 1e-12) or np.any(xy[:, 1] > 1 - r + 1e-12):
                    boundary_ok = False
                if boundary_ok:
                    break

            # Apply corrections
            correction = np.zeros((n, 2))
            for i, j in zip(iu[0], iu[1]):
                d = dist[i, j]
                if d < 1e-15:
                    angle = rng.uniform(0, 2 * np.pi)
                    fx = np.cos(angle)
                    fy = np.sin(angle)
                    overlap = min_dist[i, j]
                else:
                    fx = (xy[j, 0] - xy[i, 0]) / d
                    fy = (xy[j, 1] - xy[i, 1]) / d
                    overlap = min_dist[i, j] - d

                if overlap > 0:
                    wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                    wj = 1 - wi
                    correction[i] += np.array([-fx * overlap * wi, -fy * overlap * wi])
                    correction[j] += np.array([fx * overlap * wj, fy * overlap * wj])

            xy += correction

            # Boundary projection
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)

        return np.concatenate([xy.flatten(), r])

    def gradient_ascent_radii(xy, n_iters=200, tol=1e-15):
        """Increase radii while maintaining constraints via iterative adjustment."""
        n = N
        r = np.full(n, 0.03)  # Start with small radii
        xy = xy.copy()

        for it in range(n_iters):
            improved = False
            for i in range(n):
                # Try to increase radius of circle i
                max_possible_r = min(
                    xy[i, 0], 1 - xy[i, 0],
                    xy[i, 1], 1 - xy[i, 1]
                )
                # Distance to other circles
                min_dist_to_other = max_possible_r
                for j in range(n):
                    if i == j:
                        continue
                    d = np.sqrt((xy[i, 0] - xy[j, 0]) ** 2 + (xy[i, 1] - xy[j, 1]) ** 2)
                    min_dist_to_other = min(min_dist_to_other, d - r[j])

                # Can we increase r[i]?
                new_r = min(max_possible_r, min_dist_to_other)
                if new_r > r[i] + tol:
                    r[i] = new_r
                    improved = True

            if not improved:
                # Try to move circles to allow larger radii
                break

        return xy, r

    def get_initial_config(restart, rng):
        n = N

        if restart % 12 == 0:
            # Grid packing (6x6, use 32)
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
                r[i] = 0.5 / side - 0.002
        elif restart % 12 == 1:
            # Perturbed grid
            side = 6
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
                r[i] = 0.5 / side - 0.002
            xy += rng.uniform(-0.015, 0.015, size=(n, 2))
        elif restart % 12 == 2:
            # Hexagonal packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            cols_per_row = [6, 5, 6, 5, 6, 4]  # 32 total
            idx = 0
            for row, n_cols in enumerate(cols_per_row):
                offset = 0.5 / 6 if row % 2 == 1 else 0
                for col in range(n_cols):
                    x = (col + 0.5) / 6 + offset
                    y = (row + 0.5) / 6
                    if idx < n:
                        xy[idx] = [x, y]
                        r[idx] = 0.078
                        idx += 1
        elif restart % 12 == 3:
            # Random uniform
            xy = rng.uniform(0.06, 0.94, size=(n, 2))
            r = rng.uniform(0.05, 0.09, size=n)
        elif restart % 12 == 4:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5) / 8, (row + 0.5) / 4]
                r[i] = 0.5 / 8 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif restart % 12 == 5:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5) / 4, (row + 0.5) / 8]
                r[i] = 0.5 / 4 - 0.002
            xy += rng.uniform(-0.008, 0.008, size=(n, 2))
        elif restart % 12 == 6:
            # Hexagonal variant with different spacing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5) / 6 + (0.15 if row % 2 == 0 else 0)
                y = (row + 0.5) / 6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 12 == 7:
            # Clustered: 4 groups of 8
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5 / 3
                sy = (sub_i // 3) / 3.0 + 0.5 / 3
                xy[i] = [cx + (sx - 0.5) * 0.35, cy + (sy - 0.5) * 0.35]
                r[i] = 0.055
        elif restart % 12 == 8:
            # Random medium
            xy = rng.uniform(0.06, 0.94, size=(n, 2))
            r = rng.uniform(0.06, 0.085, size=n)
        elif restart % 12 == 9:
            # Random small
            xy = rng.uniform(0.04, 0.96, size=(n, 2))
            r = rng.uniform(0.03, 0.055, size=n)
        elif restart % 12 == 10:
            # 5x7 grid (35 cells, use 32)
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 7
                col = i % 7
                xy[i] = [(col + 0.5) / 7, (row + 0.5) / 5]
                r[i] = min(0.5 / 7, 0.5 / 5) - 0.002
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        else:
            # 7x5 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 5
                col = i % 5
                xy[i] = [(col + 0.5) / 5, (row + 0.5) / 7]
                r[i] = min(0.5 / 5, 0.5 / 7) - 0.002
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))

        params = np.concatenate([xy.flatten(), r])
        return params

    def refine_solution(params, max_iter=400, tol=1e-14):
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def high_precision_polish(params, max_iter=3000, tol=1e-16):
        """Dedicated high-precision refinement with tighter tolerances."""
        try:
            result = minimize(penalty_function, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-14,
                                      'maxls': 80})
            return result.x
        except Exception:
            return params

    def voronoi_based_packing(rng):
        """
        Start with a grid, then iteratively adjust centers and radii based on local clearance.
        """
        n = N
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)

        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.07

        # Add some randomness
        xy += rng.uniform(-0.025, 0.025, size=(n, 2))
        r += rng.uniform(-0.008, 0.008, size=n)
        r = np.clip(r, 0.03, 0.12)

        params = np.concatenate([xy.flatten(), r])
        params = push_apart_vectorized(params, n_iters=60, rng=rng)

        # Refine
        refined = refine_solution(params, max_iter=350, tol=1e-14)

        if is_valid(refined):
            return refined
        else:
            repaired = push_apart_vectorized(refined, n_iters=80, rng=rng)
            if is_valid(repaired):
                return repaired
            else:
                return params if is_valid(params) else None

    rng = np.random.RandomState(42)

    # Phase 1: Many restarts with diverse initializations
    n_restarts = 120
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break

        # Use Voronoi-based approach for some restarts
        if restart % 6 == 0:
            params = voronoi_based_packing(rng)
            if params is None:
                continue
        else:
            params = get_initial_config(restart, rng)
            params = push_apart_vectorized(params, n_iters=40, rng=rng)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        # Refine with L-BFGS-B
        refined = refine_solution(params, max_iter=350, tol=1e-13)

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Try to repair
            repaired = push_apart_vectorized(refined, n_iters=60, rng=rng)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink radii until valid
                xy = refined[:2 * N].reshape(N, 2).copy()
                r = refined[2 * N:].copy()
                for _ in range(150):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.94
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 15:
        # First: high-precision polish on the current best
        polished = high_precision_polish(best_result, max_iter=2500, tol=1e-16)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Second: multiple perturbation attempts with varying scales
        for polish_iter in range(25):
            if time.time() > deadline - 8:
                break
            # Vary perturbation scale: start small, occasionally try larger
            if polish_iter < 12:
                scale = 0.0015
            elif polish_iter < 18:
                scale = 0.004
            else:
                scale = 0.008

            perturbed = best_result.copy()
            xy_p = perturbed[:2 * N].reshape(N, 2).copy()
            r_p = perturbed[2 * N:].copy()
            xy_p += rng.uniform(-scale, scale, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart_vectorized(perturbed, n_iters=35, rng=rng)
            polished2 = high_precision_polish(perturbed, max_iter=1200, tol=1e-15)
            if is_valid(polished2):
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

        # Third: try radius perturbation as well
        if time.time() < deadline - 8:
            for r_pert_iter in range(8):
                if time.time() > deadline - 8:
                    break
                perturbed = best_result.copy()
                xy_p = perturbed[:2 * N].reshape(N, 2).copy()
                r_p = perturbed[2 * N:].copy()
                r_p *= (1 + rng.uniform(-0.015, 0.015, size=N))
                r_p = np.clip(r_p, 0.01, 0.3)
                perturbed = np.concatenate([xy_p.flatten(), r_p])
                perturbed = push_apart_vectorized(perturbed, n_iters=50, rng=rng)
                polished3 = high_precision_polish(perturbed, max_iter=1200, tol=1e-15)
                if is_valid(polished3):
                    score = get_score(polished3)
                    if score > best_score:
                        best_score = score
                        best_result = polished3.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2 * n].reshape(n, 2).copy()
        r = best_result[2 * n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            # Final micro-polish
            if time.time() < deadline - 3:
                final_polished = high_precision_polish(final_params, max_iter=800, tol=1e-16)
                if is_valid(final_polished):
                    score = get_score(final_polished)
                    if score > get_score(final_params):
                        final_params = final_polished
                        xy = final_params[:2 * n].reshape(n, 2).copy()
                        r = final_params[2 * n:].copy()
                        r = np.clip(r, 0, 0.5)
                        xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
                        xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            return np.column_stack([xy, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
