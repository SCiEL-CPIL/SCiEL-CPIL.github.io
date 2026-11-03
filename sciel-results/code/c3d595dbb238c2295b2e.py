import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def compute_radius(centers):
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        # Use cKDTree for efficiency
        tree = cKDTree(centers)
        d_min_arr, _ = tree.query(centers, k=2)
        min_pair = np.min(d_min_arr[:, 1])
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

    def get_bottleneck(centers):
        """Find the bottleneck: min of wall distances and half pair distances.
        Returns (bottleneck_value, type, indices).
        type: 'wall' or 'pair'
        """
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        min_wall_val = wall_dists.flat[min_wall_idx]
        wall_circle = min_wall_idx // 2
        wall_axis = min_wall_idx % 2

        tree = cKDTree(centers)
        d_arr, idx_arr = tree.query(centers, k=2)
        min_pair_idx = np.argmin(d_arr[:, 1])
        min_pair_val = d_arr[min_pair_idx, 1] / 2.0
        pair_i = min_pair_idx
        pair_j = idx_arr[min_pair_idx, 1]

        if min_wall_val < min_pair_val:
            return min_wall_val, 'wall', (wall_circle, wall_axis)
        else:
            return min_pair_val, 'pair', (pair_i, pair_j)

    def simulated_annealing(centers, time_limit=30.0):
        """
        Simulated annealing on the exact non-smooth objective.
        Move one circle at a time, accept if radius improves or with probability exp(-delta/T).
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()

        T = 0.01  # initial temperature
        T_min = 1e-6
        cooling = 0.9995
        max_steps = 200000

        rng = np.random.default_rng(42)
        start = time.time()
        steps = 0

        while steps < max_steps:
            if time.time() > start + time_limit:
                break
            if T < T_min:
                break

            # Pick a random circle to move
            idx = rng.integers(0, N)

            # Generate a random perturbation (Gaussian)
            sigma = 0.02 * (T / 0.01)  # scale perturbation with temperature
            new_pos = centers[idx] + rng.normal(0, sigma, 2)
            new_pos = np.clip(new_pos, -0.5, 0.5)

            old_pos = centers[idx].copy()
            centers[idx] = new_pos
            new_r = compute_radius(centers)

            delta = new_r - current_r
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                centers[idx] = old_pos

            T *= cooling
            steps += 1

        return best_centers, best_r

    def bottleneck_elimination_sa(centers, time_limit=30.0):
        """
        Hybrid: identify bottleneck, then do targeted SA moves on the bottleneck
        circle(s) to eliminate it, then global SA.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()

        rng = np.random.default_rng(123)
        start = time.time()

        T = 0.005
        T_min = 1e-7
        cooling = 0.9998

        while time.time() < start + time_limit:
            if T < T_min:
                T = T_min
                break

            # Identify bottleneck
            bottleneck_val, btype, binfo = get_bottleneck(centers)

            if btype == 'wall':
                circle_idx = binfo[0]
            else:
                circle_idx = binfo[0]  # move one of the pair

            # Generate targeted perturbation for the bottleneck circle
            sigma = 0.015 * (T / 0.005)
            new_pos = centers[circle_idx] + rng.normal(0, sigma, 2)
            new_pos = np.clip(new_pos, -0.5, 0.5)

            old_pos = centers[circle_idx].copy()
            centers[circle_idx] = new_pos
            new_r = compute_radius(centers)

            delta = new_r - current_r
            if delta >= 0 or rng.random() < math.exp(delta / T):
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            else:
                centers[circle_idx] = old_pos

            # Also do a random move occasionally for exploration
            if rng.random() < 0.3:
                ridx = rng.integers(0, N)
                rnew = centers[ridx] + rng.normal(0, sigma, 2)
                rnew = np.clip(rnew, -0.5, 0.5)
                rold = centers[ridx].copy()
                centers[ridx] = rnew
                rnew_r = compute_radius(centers)
                rdelta = rnew_r - current_r
                if rdelta >= 0 or rng.random() < math.exp(rdelta / T):
                    current_r = rnew_r
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
                else:
                    centers[ridx] = rold

            T *= cooling

        return best_centers, best_r

    def fast_local_search(centers, time_limit=10.0):
        """
        Fast local search: repeatedly find the worst circle and try to move it
        to a better position using a small number of candidate directions.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()

        start = time.time()
        rng = np.random.default_rng(7)

        while time.time() < start + time_limit:
            # Find worst circle (smallest clearance)
            wall_dists = 0.5 - np.abs(centers)
            min_wall_per_circle = np.min(wall_dists, axis=1)

            tree = cKDTree(centers)
            d_min_arr, _ = tree.query(centers, k=2)
            pair_clearance_per_circle = d_min_arr[:, 1] / 2.0

            clearance_per_circle = np.minimum(min_wall_per_circle, pair_clearance_per_circle)
            worst_idx = np.argmin(clearance_per_circle)

            # Try several random moves for this circle
            improved = False
            for _ in range(20):
                sigma = 0.01 + 0.03 * rng.random()
                new_pos = centers[worst_idx] + rng.normal(0, sigma, 2)
                new_pos = np.clip(new_pos, -0.5, 0.5)

                old_pos = centers[worst_idx].copy()
                centers[worst_idx] = new_pos
                new_r = compute_radius(centers)

                if new_r > current_r + 1e-12:
                    current_r = new_r
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
                    improved = True
                    break
                else:
                    centers[worst_idx] = old_pos

            if not improved:
                # Try a random circle
                ridx = rng.integers(0, N)
                for _ in range(5):
                    sigma = 0.01 + 0.03 * rng.random()
                    new_pos = centers[ridx] + rng.normal(0, sigma, 2)
                    new_pos = np.clip(new_pos, -0.5, 0.5)

                    old_pos = centers[ridx].copy()
                    centers[ridx] = new_pos
                    new_r = compute_radius(centers)

                    if new_r > current_r + 1e-12:
                        current_r = new_r
                        if current_r > best_r:
                            best_r = current_r
                            best_centers = centers.copy()
                        break
                    else:
                        centers[ridx] = old_pos

        return best_centers, best_r

    # Main logic
    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    inits = []

    # Hexagonal grids
    for n_rows in range(13, 17):
        if time.time() > deadline - 90:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.98, 1.0, 1.02]:
            pts = hexagonal_init(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)

    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    inits.append(grid)

    # Perturbed grids
    rng_init = np.random.default_rng(99)
    for _ in range(3):
        if time.time() > deadline - 90:
            break
        perturbed = grid + rng_init.normal(0, 0.008, (N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        inits.append(perturbed)

    # Random
    for _ in range(2):
        if time.time() > deadline - 90:
            break
        rand_pts = rng_init.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_pts)

    # For each init, run the new strategy
    for init in inits:
        if time.time() > deadline - 20:
            break

        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()

        # Phase 1: Fast local search
        if time.time() < deadline - 15:
            ls_centers, ls_r = fast_local_search(init, time_limit=8.0)
            if ls_r > best_radius:
                best_radius = ls_r
                best_centers = ls_centers.copy()

        # Phase 2: Bottleneck elimination SA
        if time.time() < deadline - 10:
            sa_centers, sa_r = bottleneck_elimination_sa(best_centers, time_limit=12.0)
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()

    # Final global SA if time remains
    if time.time() < deadline - 3 and best_centers is not None:
        final_centers, final_r = simulated_annealing(best_centers, time_limit=5.0)
        if final_r > best_radius:
            best_radius = final_r
            best_centers = final_centers.copy()

    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
