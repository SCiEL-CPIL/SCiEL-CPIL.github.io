"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import time
import numpy as np
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
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1  # Fallback


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


def _refine_repulsion(centers, max_iters=50000):
    """
    Refine the packing using a repulsion-based optimizer.
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


def _refine_l_bfgs(pts, max_iter=200, t=0.01):
    """Refine a configuration using L-BFGS-B on the soft-min objective."""

    def objective_and_grad(x):
        val, grad = _soft_min_objective_and_grad(x, t)
        return val, grad

    x0 = pts.flatten()
    res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
    return res.x.reshape(-1, 2)


def construct():
    best_pts = None
    best_r = 0.0

    np.random.seed(42)

    # Initial guess: hexagonal-like grid
    pts0 = _hexagonal_grid(N)

    # First, refine with repulsion-based optimizer
    pts_rep, r_rep = _refine_repulsion(pts0, max_iters=30000)
    if r_rep > best_r:
        best_r = r_rep
        best_pts = pts_rep.copy()

    # Then, polish with L-BFGS-B
    if _remaining_time() > 10.0:
        pts_opt = _refine_l_bfgs(best_pts, max_iter=300)
        r = _compute_radius(pts_opt)
        if r > best_r:
            best_r = r
            best_pts = pts_opt.copy()

    # Random restarts
    max_restarts = 20
    for i in range(max_restarts):
        if _remaining_time() < 5.0:
            break

        # Generate a random initial guess based on the grid
        noise_scale = 0.005 + 0.01 * (i % 5)
        pts_rand = _hexagonal_grid(N) + np.random.randn(N, 2) * noise_scale
        pts_rand[:, 0] = np.clip(pts_rand[:, 0], -0.49, 0.49)
        pts_rand[:, 1] = np.clip(pts_rand[:, 1], -0.49, 0.49)

        # Refine with repulsion
        pts_rep, r_rep = _refine_repulsion(pts_rand, max_iters=15000)

        # Polish with L-BFGS-B if time permits
        if _remaining_time() > 5.0:
            pts_opt = _refine_l_bfgs(pts_rep, max_iter=100)
            r = _compute_radius(pts_opt)
            if r > r_rep:
                pts_rep = pts_opt
                r_rep = r

        if r_rep > best_r:
            best_r = r_rep
            best_pts = pts_rep.copy()

    # Final polish on the best solution
    if best_pts is not None and _remaining_time() > 3.0:
        pts_polish = _refine_l_bfgs(best_pts, max_iter=200)
        r = _compute_radius(pts_polish)
        if r > best_r:
            best_r = r
            best_pts = pts_polish.copy()

    return best_pts
