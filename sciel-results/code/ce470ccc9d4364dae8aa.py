"""Packing 200 equal circles in a unit square.
Combines hexagonal initialization, L-BFGS-B optimization with soft-min objective,
and bottleneck-driven polishing to escape local minima.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    wall_min = min(np.min(dx), np.min(dy))
    if wall_min < 0:
        return 0.0
    # Pairwise distances
    dists = pdist(centers)
    pair_min = np.min(dists) / 2.0
    return min(wall_min, pair_min)


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # k = i * (2n - i - 1) // 2 + j - i - 1
    # For n=200, use a fast approach
    i = int(np.ceil((n - 1 - np.sqrt(1 + 8.0 * k)) / 2.0))
    if i < 0:
        i = 0
    start = i * (2 * n - i - 1) // 2
    j = i + 1 + (k - start)
    return i, j


def _hexagonal_grid(n_circles):
    """
    Create a hexagonal packing arrangement for n_circles circles.
    Returns an array of shape (n_circles, 2).
    """
    best_centers = None
    best_r = 0.0

    # Try different numbers of rows
    for m in range(10, 25):
        base_per_row = n_circles // m
        extra = n_circles % m

        row_counts = []
        for i in range(m):
            if i < extra:
                row_counts.append(base_per_row + 1)
            else:
                row_counts.append(base_per_row)

        max_row_count = max(row_counts)

        r_width = 1.0 / (2.0 * max_row_count)
        r_height = 1.0 / ((m - 1) * np.sqrt(3) + 2)
        r_est = min(r_width, r_height)

        if r_est <= best_r and best_centers is not None:
            continue

        dx = 2.0 * r_est
        dy = np.sqrt(3) * r_est

        centers = []
        y_offset = -0.5 + r_est

        for row_idx, row_count in enumerate(row_counts):
            x_offset = 0.0
            if row_idx % 2 == 1:
                x_offset = dx / 2.0

            if row_count > 0:
                row_width = (row_count - 1) * dx
                start_x = -row_width / 2.0 + x_offset
            else:
                start_x = 0.0

            for col_idx in range(row_count):
                x = start_x + col_idx * dx
                y = y_offset + row_idx * dy
                centers.append([x, y])

        centers = np.array(centers)

        # Center the arrangement in the square
        centers[:, 0] -= np.mean(centers[:, 0])
        centers[:, 1] -= np.mean(centers[:, 1])

        # Clip to be within bounds
        centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)

        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (n_circles, 2))


def _soft_min_objective_and_grad(x, t=0.01):
    """
    Compute the soft-min objective and its gradient.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)

    # Wall distances
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dists = np.concatenate([dx, dy])

    # Pairwise distances / 2
    dists = pdist(pts)
    pair_dists = dists / 2.0

    all_dists = np.concatenate([wall_dists, pair_dists])

    # Soft min
    m = np.max(all_dists)
    exp_vals = np.exp(-(all_dists - m) / t)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    soft_min_val = m - t * np.log(sum_exp)

    # Gradient of soft_min with respect to pts
    grad = np.zeros_like(pts)

    # Wall gradients
    gx_wall = -np.sign(pts[:, 0])
    gy_wall = -np.sign(pts[:, 1])

    grad[:, 0] += weights[:N] * gx_wall
    grad[:, 1] += weights[N:2 * N] * gy_wall

    # Pairwise gradients
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)

    weights_pair = weights[2 * N:2 * N + num_pairs]

    diffs = pts[i_idx] - pts[j_idx]  # shape (num_pairs, 2)
    dists_matrix = dists  # shape (num_pairs,)

    safe_dists = np.where(dists_matrix > 1e-10, dists_matrix, 1.0)

    terms = weights_pair[:, np.newaxis] * diffs / (2.0 * safe_dists[:, np.newaxis])

    np.add.at(grad, i_idx, terms)
    np.add.at(grad, j_idx, -terms)

    obj_val = -soft_min_val
    grad_val = -grad.flatten()

    return obj_val, grad_val


def _refine_lbfgs(pts, max_iter=500, t=0.01):
    """Refine a configuration using L-BFGS-B on the soft-min objective."""

    def objective_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, t)
        return val, grad

    x0 = pts.flatten()
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
    return res.x.reshape(-1, 2)


def _refine_repulsion(centers, max_iters=50000):
    """
    Refine the packing using a repulsion-based optimizer to escape local minima.
    """
    centers = centers.copy()
    r = _compute_radius(centers)
    best_r = r
    best_centers = centers.copy()

    step_size = 0.02
    no_improve = 0

    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break

        # Compute distances
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        pair_term = min_pair_dist / 2.0

        dw = np.minimum(np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
                        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1]))
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        wall_term = min_wall_dist

        current_r = min(pair_term, wall_term)

        if pair_term <= wall_term:
            # Pair is the bottleneck
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Wall is the bottleneck
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move

        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)

        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                # Random perturbation
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.015
                centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                no_improve = 0
                step_size = max(0.002, step_size * 0.92)
            elif no_improve > 100:
                step_size = max(0.002, step_size * 0.97)

    return best_centers, best_r


def construct():
    """Construct optimal circle packing."""
    np.random.seed(42)

    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()

    # Phase 1: L-BFGS-B refinement
    if _remaining_time() > 10.0:
        try:
            refined = _refine_lbfgs(centers, max_iter=300, t=0.02)
            refined = np.clip(refined, -0.5, 0.5)
            r_refined = _compute_radius(refined)
            if r_refined > best_radius:
                best_radius = r_refined
                best_centers = refined.copy()
        except Exception:
            pass

    # Phase 2: Repulsion-based refinement to escape local minima
    if _remaining_time() > 5.0:
        try:
            optimized, opt_radius = _refine_repulsion(best_centers, max_iters=30000)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized.copy()
        except Exception:
            pass

    # Phase 3: Another L-BFGS-B pass with smaller t for precision
    if _remaining_time() > 5.0:
        try:
            refined = _refine_lbfgs(best_centers, max_iter=200, t=0.005)
            refined = np.clip(refined, -0.5, 0.5)
            r_refined = _compute_radius(refined)
            if r_refined > best_radius:
                best_radius = r_refined
                best_centers = refined.copy()
        except Exception:
            pass

    # Phase 4: Final short repulsion pass
    if _remaining_time() > 3.0:
        try:
            optimized, opt_radius = _refine_repulsion(best_centers, max_iters=10000)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized.copy()
        except Exception:
            pass

    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers
