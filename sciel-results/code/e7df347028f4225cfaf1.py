"""Force-directed layout with gradient polishing for 50 equal circles in a unit square."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the true maximum radius for a given configuration."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_dists_x, wall_dists_y))

    if min_wall < 0:
        return 0.0

    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0

    return min(min_pair, min_wall)


def _soft_min_objective(x, tau=0.005):
    """
    Objective: minimize -soft_min(minimum distance to constraint boundary).
    Uses full pairwise distances for accuracy.
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Full pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0

    # Upper triangle pairwise distances
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]

    all_dists = np.concatenate([walls, pair_upper])

    # Soft-min using log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum

    return -soft_min


def _grad_soft_min(x, tau=0.005):
    """
    Compute gradient of -soft_min objective analytically.
    """
    centers = x.reshape(-1, 2)

    # Wall distances
    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    # Full pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0

    # Upper triangle
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]

    all_dists = np.concatenate([walls, pair_upper])

    # Soft-min weights
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)

    grad = np.zeros_like(x)

    # Wall gradient contributions
    for i in range(N):
        w_wall = weights[i]
        if walls[i] <= wall_dists_x[i] + 1e-15:
            grad[2*i] += w_wall * np.sign(centers[i, 0])
        if walls[i] <= wall_dists_y[i] + 1e-15:
            grad[2*i+1] += w_wall * np.sign(centers[i, 1])

    # Pairwise gradient contributions
    for k in range(len(iu)):
        i, j = iu[k], ju[k]
        w_pair = weights[N + k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w_pair / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij

    return grad


def _force_directed_layout(centers, time_budget, rng):
    """
    Force-directed layout: repulsive forces between circles, attractive toward center.
    Uses adaptive step size and cooling.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    n_steps = 50000
    # Adaptive parameters
    repulsion_power = 3.0  # Force ~ 1/d^power
    attraction_strength = 0.1
    base_step = 0.005

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Compute forces
        forces = np.zeros_like(current)

        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-9:
                    # Random perturbation for coincident points
                    diff = rng.normal(size=2)
                    d = np.linalg.norm(diff)
                unit = diff / d
                # Repulsive force: stronger when closer
                force_mag = repulsion_strength / (d ** repulsion_power) if d < 0.15 else 0.0
                forces[i] += unit * force_mag
                forces[j] -= unit * force_mag

        # Attraction toward center (weak)
        for i in range(N):
            forces[i] -= attraction_strength * current[i]

        # Apply forces with adaptive step
        step_size = base_step * (1.0 - 0.5 * step / n_steps)
        trial = current + forces * step_size
        trial = np.clip(trial, -0.5, 0.5)

        # Evaluate
        trial_r = _compute_radius(trial)

        if trial_r > best_r:
            best_r = trial_r
            best = trial.copy()

        # Accept if improvement or with probability
        if trial_r >= _compute_radius(current) or (trial_r > 0 and rng.random() < 0.1):
            current = trial

    return best, best_r


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing to optimize the minimum radius directly.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    T_start = 0.012
    T_end = 0.00003
    n_steps = 40000
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Compute current distances
        tree = cKDTree(current)
        dists, idx_nn = tree.query(current, k=2)
        wall_x = 0.5 - np.abs(current[:, 0])
        wall_y = 0.5 - np.abs(current[:, 1])
        min_pair = np.min(dists[:, 1]) / 2.0
        min_wall = np.min(np.minimum(wall_x, wall_y))
        curr_r = min(min_pair, min_wall)

        # Weighted selection of constraint to improve
        all_dists = np.concatenate([np.minimum(wall_x, wall_y), dists[:, 1] / 2.0])
        min_d = np.min(all_dists)
        weights = np.exp(-(all_dists - min_d) / 0.003)
        weights /= np.sum(weights)

        idx = rng.choice(len(all_dists), p=weights)

        move_size = 0.002 + 0.006 * (T / T_start)

        if idx < N:
            i = idx
            if wall_x[i] <= wall_y[i]:
                delta = np.array([np.sign(current[i, 0]) if current[i, 0] != 0 else rng.choice([-1, 1]), 0.0])
            else:
                delta = np.array([0.0, np.sign(current[i, 1]) if current[i, 1] != 0 else rng.choice([-1, 1])])
            delta += rng.normal(0, 0.001, 2)
            trial = current.copy()
            trial[i] += delta * move_size
            trial = np.clip(trial, -0.5, 0.5)
        else:
            i = idx - N
            j = idx_nn[i, 1]
            diff = current[i] - current[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-9:
                unit = diff / dist
            else:
                unit = rng.normal(size=2)
                unit /= np.linalg.norm(unit)
            trial = current.copy()
            trial[i] += unit * move_size * rng.uniform(0.3, 0.8)
            trial[j] -= unit * move_size * rng.uniform(0.3, 0.8)
            trial[i] += rng.normal(0, 0.0008, 2)
            trial[j] += rng.normal(0, 0.0008, 2)
            trial = np.clip(trial, -0.5, 0.5)

        trial_r = _compute_radius(trial)
        delta_r = trial_r - curr_r

        if delta_r >= 0 or rng.random() < np.exp(delta_r / max(T, 1e-10)):
            current = trial
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()

        T *= cooling

    return best, best_r


def _lbfgs_optimize(x0, tau=0.005, time_budget=15.0):
    """Run L-BFGS-B optimization on the soft-min objective."""
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_objective(x, tau)
        g = _grad_soft_min(x, tau)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxfun': 2000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _high_precision_polish(centers, time_budget):
    """
    High-precision polish using L-BFGS with progressively smaller tau.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    for tau in [0.003, 0.0015, 0.0008, 0.0004]:
        if time.time() - start_time > time_budget:
            break

        x0 = best.flatten().copy()

        def obj_and_grad(x):
            if time.time() - start_time > time_budget:
                return 0.0, np.zeros_like(x)
            f = _soft_min_objective(x, tau)
            g = _grad_soft_min(x, tau)
            return f, g

        try:
            result = opt.minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxfun': 5000, 'disp': False}
            )
            polished = result.x.reshape(-1, 2)
            polished = np.clip(polished, -0.5, 0.5)
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_r = r_polished
                best = polished
        except Exception:
            pass

    return best, best_r


def _exact_local_polish(centers, time_budget):
    """
    Direct local optimization: identify bottleneck constraints and
    perform focused coordinate ascent.
    """
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    rng_local = np.random.default_rng(42)

    for iteration in range(80):
        if time.time() - start_time > time_budget:
            break

        c = best.copy()
        r = _compute_radius(c)

        # Compute all constraint distances
        tree = cKDTree(c)
        dists, idx_nn = tree.query(c, k=2)
        pair_dists = dists[:, 1] / 2.0
        wall_x = 0.5 - np.abs(c[:, 0])
        wall_y = 0.5 - np.abs(c[:, 1])
        wall_dists = np.minimum(wall_x, wall_y)

        all_dists = np.concatenate([wall_dists, pair_dists])
        min_r = np.min(all_dists)

        tol = 1e-7
        active_wall = np.where(wall_dists < min_r + tol)[0]
        active_pair = np.where(pair_dists < min_r + tol)[0]

        if len(active_wall) == 0 and len(active_pair) == 0:
            break

        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break

            my_wall = wall_dists[i]
            my_pair = pair_dists[i]
            my_min = min(my_wall, my_pair)

            if my_min > min_r + tol:
                continue

            for _ in range(30):
                if time.time() - start_time > time_budget:
                    break

                direction = rng_local.normal(size=2)
                direction /= np.linalg.norm(direction) + 1e-15

                for step in [0.0008, 0.0004, 0.0002, 0.0001]:
                    trial = c.copy()
                    trial[i] += direction * step
                    trial = np.clip(trial, -0.5, 0.5)

                    trial_r = _compute_radius(trial)
                    if trial_r > best_r + 1e-13:
                        best = trial
                        best_r = trial_r
                        c = trial
                        improved = True
                        break
                    elif trial_r >= min_r - 1e-10:
                        c = trial

        if not improved:
            break

    return best, best_r


def _initial_hex_grid():
    """Create initial hexagonal-like layout."""
    # Hexagonal packing in a square
    # Use a grid with offset rows
    rows = 8
    cols = 7
    spacing_x = 1.0 / (cols + 0.5)
    spacing_y = 1.0 / (rows + 0.5)

    points = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) * spacing_x - 0.5
            y = (row + 0.5) * spacing_y - 0.5
            if row % 2 == 1:
                x += spacing_x / 2
            points.append([x, y])

    # Add more points to reach N
    while len(points) < N:
        # Add points in gaps
        idx = len(points) % (len(points) + 1)
        base = points[idx % len(points)]
        points.append([base[0] + 0.01, base[1] + 0.01])

    points = np.array(points[:N])

    rng = np.random.default_rng(12345)
    points = points + rng.normal(0, 0.003, points.shape)
    points = np.clip(points, -0.49, 0.49)

    return points


def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.004, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration: try hex grid
    best_centers = _initial_hex_grid()
    best_radius = _compute_radius(best_centers)

    # Also try standard grid
    grid_centers = _initial_grid()
    grid_radius = _compute_radius(grid_centers)
    if grid_radius > best_radius:
        best_centers = grid_centers
        best_radius = grid_radius

    rng = np.random.default_rng(777)

    # Phase 1: Force-directed layout
    fd_budget = min(35.0, (deadline - time.time()) * 0.3)
    if fd_budget > 5:
        trial_centers, trial_radius = _force_directed_layout(best_centers, fd_budget, rng)
        if trial_radius > best_radius:
            best_radius = trial_radius
            best_centers = trial_centers

    # Phase 2: Simulated Annealing
    sa_budget = min(40.0, (deadline - time.time()) * 0.35)
    if sa_budget > 5:
        trial_centers, trial_radius = _simulated_annealing(best_centers, rng, sa_budget)
        if trial_radius > best_radius:
            best_radius = trial_radius
            best_centers = trial_centers

    # Phase 3: L-BFGS optimization with soft-min
    if time.time() < deadline - 40:
        x0 = best_centers.flatten()
        for tau in [0.008, 0.005, 0.003]:
            if time.time() > deadline - 25:
                break
            trial = _lbfgs_optimize(x0, tau=tau, time_budget=min(12, deadline - time.time() - 18))
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r
                best_centers = trial
            x0 = best_centers.flatten()

    # Phase 4: High-precision polish
    if time.time() < deadline - 25:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20, deadline - time.time() - 12)
        )

    # Phase 5: Exact local polish
    if time.time() < deadline - 12:
        best_centers, best_radius = _exact_local_polish(
            best_centers, time_budget=min(10, deadline - time.time() - 5)
        )

    # Final validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    return best_centers
