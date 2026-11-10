"""Program for 200 equal circles in a unit square."""
import numpy as np
import time

N = 200
TIME_LIMIT = 145.0  # seconds, leaving margin before 150s kill


def _compute_radius(centres):
    """Compute the packing radius for a set of centres."""
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    wall_dists = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall <= 0:
        return 0.0
    from scipy.spatial.distance import cdist
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    return min(min_dist / 2.0, min_wall)


def _grid_packing():
    """Create a square grid packing with optimal dimensions for N points."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hexagonal_packing():
    """Create a hexagonal packing pattern, optimized for N points."""
    best_centres = None
    best_r = -1

    for n_rows in range(10, 25):
        for start_offset in [0, 1]:
            for n_cols in range(10, 25):
                total_points = 0
                for row in range(n_rows):
                    if (row + start_offset) % 2 == 0:
                        total_points += n_cols
                    else:
                        total_points += n_cols - 1

                if total_points < N:
                    continue

                s_horiz = 1.0 / n_cols
                if n_rows > 1:
                    s_vert = 2.0 / ((n_rows - 1) * np.sqrt(3))
                else:
                    s_vert = 1.0

                s = min(s_horiz, s_vert)

                centres = []
                y_start = -0.5 + s * np.sqrt(3) / 2
                for row in range(n_rows):
                    y = y_start + row * s * np.sqrt(3) / 2
                    if (row + start_offset) % 2 == 0:
                        x_start = -0.5 + s / 2
                        for col in range(n_cols):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))
                    else:
                        x_start = -0.5 + s
                        for col in range(n_cols - 1):
                            x = x_start + col * s
                            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                                centres.append((x, y))

                centres = np.array(centres)
                if len(centres) < N:
                    continue

                centres = centres[:N]
                actual_r = _compute_radius(centres)
                if actual_r > best_r:
                    best_r = actual_r
                    best_centres = centres

    if best_centres is not None:
        return best_centres
    return _grid_packing()


def _fibonacci_lattice(n):
    """Create a Fibonacci (sunflower) lattice of n points in the unit square."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(1, n + 1, dtype=np.float64)
    r = np.sqrt(i / n)
    theta = i * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    scale = 0.48
    centres = np.column_stack([x * scale, y * scale])
    return centres


def _resolve_overlaps(centres, max_iters=200, time_budget=5.0):
    """Resolve any overlapping circles by pushing them apart."""
    from scipy.spatial.distance import cdist
    centres = centres.copy()
    start_time = time.time()
    deadline = start_time + time_budget

    for _ in range(max_iters):
        if time.time() > deadline:
            break
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist > 0.05:  # Well separated
            break

        forces = np.zeros_like(centres)
        # Push apart close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.06:  # Close threshold
                    diff = centres[i] - centres[j]
                    if d > 1e-10:
                        f = 0.01 * (0.06 / d) * diff / d
                        forces[i] += f
                        forces[j] -= f
        # Wall repulsion
        for i in range(N):
            for axis in range(2):
                pos = centres[i, axis]
                if pos < 0:
                    wall_dist = 0.5 + pos
                else:
                    wall_dist = 0.5 - pos
                if wall_dist < 0.03:
                    if pos < 0:
                        forces[i, axis] += 0.01 * (0.03 / max(wall_dist, 1e-10))
                    else:
                        forces[i, axis] -= 0.01 * (0.03 / max(wall_dist, 1e-10))

        centres += forces
        centres = np.clip(centres, -0.49, 0.49)

    return centres


def _l_bfgsb_optimize(centres, time_budget=120.0):
    """Optimize using L-BFGS-B with a smooth potential maximizing minimum distance."""
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    start_time = time.time()
    deadline = start_time + time_budget

    def objective(x):
        if time.time() > deadline:
            return 1e10
        c = x.reshape(N, 2)
        if np.any(c < -0.5) or np.any(c > 0.5):
            return 1e10
        # Wall penalty
        wall_dists = np.min(np.stack([
            0.5 - c[:, 0], 0.5 + c[:, 0],
            0.5 - c[:, 1], 0.5 + c[:, 1]
        ], axis=1), axis=1)
        min_wall = np.min(wall_dists)
        if min_wall <= 0:
            return 1e10
        # Pairwise distances
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        # Use log-sum-exp to approximate max of (1/d) which minimizes min distance
        # We want to maximize min distance, so minimize -log(min_dist)
        # Use soft approximation: -log(sum(exp(-d/T))) * T
        T = 0.01  # Temperature for soft approximation
        # Clip to avoid overflow
        d_clipped = np.clip(dists, 1e-6, None)
        neg_d_over_T = -d_clipped / T
        neg_d_over_T = neg_d_over_T - np.max(neg_d_over_T)  # For numerical stability
        log_sum = np.log(np.sum(np.exp(neg_d_over_T)))
        min_dist_soft = T * (np.max(neg_d_over_T) + log_sum)
        # Objective: minimize -(min_dist_soft) + wall penalty
        wall_penalty = 0 if min_wall > 0.001 else 1e6 * (0.001 - min_wall)
        return -min_dist_soft + wall_penalty

    def gradient(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        c = x.reshape(N, 2)
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        d_clipped = np.clip(dists, 1e-6, None)
        T = 0.01
        neg_d_over_T = -d_clipped / T
        neg_d_over_T = neg_d_over_T - np.max(neg_d_over_T)
        exp_vals = np.exp(neg_d_over_T)
        sum_exp = np.sum(exp_vals)
        # Gradient of -T * (max + log(sum(exp))) w.r.t. positions
        # d/dx_i of -min_dist_soft = sum_j (exp(-d_ij/T) / sum_k exp(-d_ik/T)) * (1/T) * (x_i - x_j)/d_ij
        grad = np.zeros_like(c)
        for i in range(N):
            weights = exp_vals[i] / sum_exp
            for j in range(N):
                if i == j:
                    continue
                d = d_clipped[i, j]
                diff = c[i] - c[j]
                grad[i] += weights[j] * (1.0 / T) * diff / d
        return grad.flatten()

    x0 = centres.flatten()
    bounds = [(-0.49, 0.49)] * (2 * N)

    try:
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            new_centres = result.x.reshape(N, 2)
            new_centres = np.clip(new_centres, -0.49, 0.49)
            return new_centres
    except Exception:
        pass

    return centres


def _local_optimization(centres, max_iters=200, step_size=0.005):
    """Local optimization using gradient-free method with constraint handling."""
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)

    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 5)

    from scipy.spatial.distance import cdist

    for it in range(max_iters):
        if time.time() > deadline:
            break

        wall_dists = np.min(np.stack([
            0.5 - best_centres[:, 0],
            0.5 + best_centres[:, 0],
            0.5 - best_centres[:, 1],
            0.5 + best_centres[:, 1]
        ], axis=1), axis=1)

        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_pair_dists = np.min(dists, axis=1)

        constraints = np.minimum(min_pair_dists / 2.0, wall_dists)
        idx = np.argmin(constraints)

        improved = False
        n_tries = 15
        for _ in range(n_tries):
            if time.time() > deadline:
                break

            angle = np.random.uniform(0, 2 * np.pi)
            delta = step_size * np.array([np.cos(angle), np.sin(angle)])
            new_centres = best_centres.copy()
            new_centres[idx] += delta

            if np.any(new_centres < -0.5) or np.any(new_centres > 0.5):
                continue

            new_r = _compute_radius(new_centres)
            if new_r > best_r + 1e-12:
                best_r = new_r
                best_centres = new_centres
                improved = True
                step_size *= 0.97
                break

        if not improved:
            step_size *= 0.92
            if step_size < 1e-7:
                break

    return best_centres, best_r


def construct():
    start_time = time.time()

    # Generate multiple initializations
    initializations = []

    # 1. Hexagonal packing (usually best for dense packings)
    hex_centres = _hexagonal_packing()
    hex_r = _compute_radius(hex_centres)
    initializations.append((hex_centres, hex_r))

    # 2. Grid packing
    grid_centres = _grid_packing()
    grid_r = _compute_radius(grid_centres)
    initializations.append((grid_centres, grid_r))

    # 3. Fibonacci lattice
    fib_centres = _fibonacci_lattice(N)
    fib_r = _compute_radius(fib_centres)
    initializations.append((fib_centres, fib_r))

    # Pick the best initialization
    initializations.sort(key=lambda x: x[1], reverse=True)
    centres, current_r = initializations[0]

    # Resolve any overlaps
    if time.time() - start_time < TIME_LIMIT - 30:
        centres = _resolve_overlaps(centres, time_budget=5.0)
        current_r = _compute_radius(centres)

    # L-BFGS-B optimization
    if time.time() - start_time < TIME_LIMIT - 15:
        time_budget = min(TIME_LIMIT - 20, 100.0)
        centres = _l_bfgsb_optimize(centres, time_budget=time_budget)
        current_r = _compute_radius(centres)

    # Local optimization (fine-tuning)
    if time.time() - start_time < TIME_LIMIT - 5:
        centres, current_r = _local_optimization(centres, max_iters=100)

    # Final check
    final_r = _compute_radius(centres)
    if final_r <= 0:
        # Fallback to a valid grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return centres
