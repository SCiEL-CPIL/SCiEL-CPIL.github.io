"""Pack 200 equal circles in the unit square [-1/2, 1/2]^2 to maximize common radius."""
import numpy as np
import time
from scipy.spatial.distance import pdist

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
    if N < 2:
        return float(wall_min)
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    return min(float(wall_min), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    # k = i * n - i * (i + 1) // 2 + j - i - 1
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
        base_per_row = N // m
        extra = N % m

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

    return best_centers if best_centers is not None else np.random.uniform(-0.5, 0.5, (N, 2))


def _refine_repulsion(centers, max_iters=200000):
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


def construct():
    """Construct optimal circle packing."""
    # Start with hexagonal grid
    centers = _hexagonal_grid(N)
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()

    # Refine with repulsion optimizer
    refined_centers, refined_radius = _refine_repulsion(centers)
    if refined_radius > best_radius:
        best_radius = refined_radius
        best_centers = refined_centers.copy()

    # Final safety clip
    best_centers = np.clip(best_centers, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_centers
