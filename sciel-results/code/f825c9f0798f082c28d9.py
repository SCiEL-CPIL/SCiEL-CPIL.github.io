import numpy as np
import time
import math
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_init(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def square_grid_init(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def best_fit_move(centers, idx, n_candidates=30):
        """
        Find a better position for circle idx by searching candidate positions
        that maximize the minimum clearance to all other circles and walls.
        Uses a k-d tree for efficient distance queries.
        """
        others = np.delete(centers, idx, axis=0)
        tree = cKDTree(others)
        current_pos = centers[idx]

        # Current clearance
        if len(others) > 0:
            d_min, _ = tree.query(current_pos)
            pair_clearance = d_min / 2.0
        else:
            pair_clearance = 1.0
        wall_clearance = 0.5 - np.abs(current_pos).max()
        current_clearance = min(pair_clearance, wall_clearance)

        # Generate candidate positions
        candidates = []
        # Radial candidates around current position
        for angle in np.linspace(0, 2 * np.pi, n_candidates, endpoint=False):
            for radius in [0.01, 0.02, 0.03, 0.04, 0.05]:
                nx = current_pos[0] + radius * math.cos(angle)
                ny = current_pos[1] + radius * math.sin(angle)
                if -0.5 <= nx <= 0.5 and -0.5 <= ny <= 0.5:
                    candidates.append([nx, ny])
        # Also try moving toward center
        candidates.append([0.0, 0.0])
        # Try small random perturbations
        for _ in range(10):
            cand = current_pos + np.random.randn(2) * 0.02
            if np.all(np.abs(cand) <= 0.5):
                candidates.append(cand)

        if not candidates:
            return current_pos

        candidates = np.array(candidates)

        best_pos = current_pos
        best_clearance = current_clearance

        for cand in candidates:
            # Clamp to bounds
            cand = np.clip(cand, -0.5, 0.5)
            if len(others) > 0:
                d_min, _ = tree.query(cand)
                pair_c = d_min / 2.0
            else:
                pair_c = 1.0
            wall_c = 0.5 - np.abs(cand).max()
            clearance = min(pair_c, wall_c)
            if clearance > best_clearance:
                best_clearance = clearance
                best_pos = cand

        return best_pos

    def best_fit_packing(centers, time_limit=8.0, n_passes=5):
        """
        Iteratively improve the packing by moving the worst-constrained circle
        to its best-fit position.
        """
        centers = centers.copy()
        start = time.time()

        for _ in range(n_passes):
            if time.time() > start + time_limit:
                break

            # Find the circle with the smallest clearance
            wall_dists = 0.5 - np.abs(centers)
            min_wall_per_circle = np.min(wall_dists, axis=1)

            # Pairwise minimum clearance per circle
            tree = cKDTree(centers)
            d_min_arr, _ = tree.query(centers, k=2)
            d_min_arr = d_min_arr[:, 1]  # nearest neighbor distance
            pair_clearance_per_circle = d_min_arr / 2.0

            clearance_per_circle = np.minimum(min_wall_per_circle, pair_clearance_per_circle)
            worst_idx = np.argmin(clearance_per_circle)

            # Try to improve worst_idx
            new_pos = best_fit_move(centers, worst_idx, n_candidates=40)
            if np.linalg.norm(new_pos - centers[worst_idx]) > 1e-10:
                centers[worst_idx] = new_pos

        return centers

    def l_bfgs_polish(centers, time_limit=5.0):
        """
        Final polish using L-BFGS on a smooth approximation of the objective.
        We minimize the negative of a smooth lower bound on the radius.
        """
        def objective(x):
            c = x.reshape(-1, 2)
            wall_dists = 0.5 - np.abs(c)
            flat_wall = wall_dists.flatten()
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            flat_pair = dist_matrix.flatten()

            vals = np.concatenate([flat_wall, flat_pair / 2.0])
            # Use a large-k log-sum-exp as soft min
            k = 100.0
            neg_k_vals = -k * vals
            max_neg_k_val = np.max(neg_k_vals)
            log_sum_exp = max_neg_k_val + np.log(np.sum(np.exp(neg_k_vals - max_neg_k_val)))
            soft_min_val = -log_sum_exp / k
            return -soft_min_val

        def gradient(x):
            c = x.reshape(-1, 2)
            wall_dists = 0.5 - np.abs(c)
            flat_wall = wall_dists.flatten()
            dist_matrix = cdist(c, c)
            np.fill_diagonal(dist_matrix, np.inf)
            flat_pair = dist_matrix.flatten()

            vals = np.concatenate([flat_wall, flat_pair / 2.0])
            k = 100.0
            neg_k_vals = -k * vals
            max_neg_k_val = np.max(neg_k_vals)
            exp_vals = np.exp(neg_k_vals - max_neg_k_val)
            sum_exp = np.sum(exp_vals)
            probs = exp_vals / sum_exp

            grad_vals = -probs
            grad_centers = np.zeros_like(c)

            # Wall part
            grad_wall_flat = grad_vals[:len(flat_wall)]
            grad_wall = grad_wall_flat.reshape(N, 2)
            sign_c = np.sign(c)
            grad_centers += grad_wall * sign_c

            # Pair part
            grad_pair_flat = grad_vals[len(flat_wall):]
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)

            grad_pair_matrix = grad_pair_flat.reshape(N, N)
            factor = 0.5 * grad_pair_matrix / dists
            contrib_i = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            grad_centers += contrib_i

            return grad_centers.flatten()

        x0 = centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x.reshape(-1, 2)
        except Exception:
            return centers

    # Main logic
    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    inits = []

    # Hexagonal grids with various parameters
    for n_rows in range(13, 17):
        if time.time() > deadline - 80:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05, 1.1]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)

    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)

    # Slightly perturbed grids
    for _ in range(2):
        if time.time() > deadline - 80:
            break
        perturbed = grid + np.random.randn(N, 2) * 0.01
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)

    # Random
    for _ in range(2):
        if time.time() > deadline - 80:
            break
        rand_pts = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)

    # For each init, run best-fit packing + polish
    for init in inits:
        if time.time() > deadline - 30:
            break

        # Quick eval
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()

        # Phase 1: Best-fit packing
        if time.time() < deadline - 15:
            bf_centers = best_fit_packing(init, time_limit=6.0, n_passes=8)
            r_bf = compute_radius(bf_centers)
            if r_bf > best_radius:
                best_radius = r_bf
                best_centers = bf_centers.copy()

        # Phase 2: L-BFGS polish
        if time.time() < deadline - 5:
            cand = best_centers
            polished = l_bfgs_polish(cand, time_limit=4.0)
            r_polished = compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()

    # Final global polish if time remains
    if time.time() < deadline - 2 and best_centers is not None:
        final_polished = l_bfgs_polish(best_centers, time_limit=2.0)
        r_final = compute_radius(final_polished)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_polished.copy()

    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
