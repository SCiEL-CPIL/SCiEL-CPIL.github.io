"""
Optimized packing of 50 equal circles in a unit square.
Uses KDTree for fast SA evaluation, soft-min log-sum-exp for polishing,
and progressive refinement for high-precision final result.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 147.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])


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

    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0

    return min(min_pair, min_wall)


def _fast_min_pair_dist(centers, moved_idx=None):
    """Compute min pairwise half-distance, optionally only checking neighbors of moved point."""
    if moved_idx is None:
        diff = centers[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        return np.sqrt(np.min(dist_sq)) / 2.0
    else:
        # Only check distances from moved point to all others
        diffs = centers[moved_idx] - centers
        dist_sq = np.sum(diffs ** 2, axis=1)
        dist_sq[moved_idx] = np.inf
        # Also need to check if removing old position increases min
        # For speed, just compute min over all pairs involving moved point
        return np.sqrt(np.min(dist_sq)) / 2.0


def _soft_min_obj(x, beta):
    """Soft-min objective: minimize negative soft-min of all constraint values."""
    centers = x.reshape(-1, 2)

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0

    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    logsumexp = np.max(z) + np.log(sum_exp)
    soft_min_val = -logsumexp / beta

    return float(-soft_min_val)


def _soft_min_grad(x, beta):
    """Gradient of the negative soft-min objective."""
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_dists = dists / 2.0

    all_vals = np.concatenate([walls, half_dists])
    z = -beta * all_vals
    z = np.clip(z, -50, 50)
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp

    n_walls = N
    w_weights = weights[:n_walls]

    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)

    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]

    if len(idx_wx) > 0:
        grad[2*idx_wx] -= w_weights[idx_wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] -= w_weights[idx_wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] -= 0.5 * w_weights[idx_wt] * np.sign(centers[idx_wt, 1])

    p_weights = weights[n_walls:]
    dists_safe = np.maximum(dists, 1e-12)
    factor = p_weights / dists_safe
    contrib = factor[:, None] * diff

    np.add.at(grad, 2 * _IU[0], contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, -contrib[:, 1])

    return grad


def _soft_violation_obj(x, r_target, beta):
    """Minimize smooth violation for target radius r_target."""
    centers = x.reshape(-1, 2)

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists

    z_wall = np.clip(beta * wall_viol, -50, 50)
    z_pair = np.clip(beta * pair_viol, -50, 50)

    soft_wall = np.log1p(np.exp(z_wall)) / beta
    soft_pair = np.log1p(np.exp(z_pair)) / beta

    return float(np.sum(soft_wall) + np.sum(soft_pair))


def _soft_violation_grad(x, r_target, beta):
    """Gradient of the soft violation objective."""
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)

    wall_dists_x = 0.5 - np.abs(centers[:, 0])
    wall_dists_y = 0.5 - np.abs(centers[:, 1])
    walls = np.minimum(wall_dists_x, wall_dists_y)
    wall_viol = r_target - walls

    z_wall = np.clip(beta * wall_viol, -50, 50)
    sig_wall = 1.0 / (1.0 + np.exp(-z_wall))

    wx = wall_dists_x < wall_dists_y
    wy = wall_dists_y < wall_dists_x
    wt = ~(wx | wy)

    idx_wx = np.nonzero(wx)[0]
    idx_wy = np.nonzero(wy)[0]
    idx_wt = np.nonzero(wt)[0]

    if len(idx_wx) > 0:
        grad[2*idx_wx] += sig_wall[wx] * np.sign(centers[idx_wx, 0])
    if len(idx_wy) > 0:
        grad[2*idx_wy+1] += sig_wall[wy] * np.sign(centers[idx_wy, 1])
    if len(idx_wt) > 0:
        grad[2*idx_wt] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 0])
        grad[2*idx_wt+1] += 0.5 * sig_wall[wt] * np.sign(centers[idx_wt, 1])

    ci = centers[_IU[0]]
    cj = centers[_IU[1]]
    diff_vec = ci - cj
    dists = np.sqrt(np.sum(diff_vec ** 2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r_target - dists

    z_pair = np.clip(beta * pair_viol, -50, 50)
    sig_pair = 1.0 / (1.0 + np.exp(-z_pair))

    factor = sig_pair / dists_safe
    contrib = factor[:, None] * diff_vec

    np.add.at(grad, 2 * _IU[0], -contrib[:, 0])
    np.add.at(grad, 2 * _IU[0] + 1, -contrib[:, 1])
    np.add.at(grad, 2 * _IU[1], contrib[:, 0])
    np.add.at(grad, 2 * _IU[1] + 1, contrib[:, 1])

    return grad


def _initial_grid():
    """Create initial grid layout with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()

    rng = np.random.default_rng(12345)
    grid = grid + rng.normal(0, 0.005, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)

    return grid


def _hex_init():
    """Create a hexagonal packing initial layout."""
    best_grid = None
    best_r = 0

    rng = np.random.default_rng(42)

    for attempt in range(12):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))

        s_width = 1.0 / (cols + 0.5)
        s_height = 1.0 / (rows * 0.866 + 0.5)
        s = min(s_width, s_height)

        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols - 1) / 2) * s
                y = (j - (rows - 1) / 2) * s * 0.866
                if j % 2 == 1:
                    x += s / 2
                centers.append([x, y])

        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])

        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_grid = centers.copy()

    return best_grid if best_grid is not None else _initial_grid()


def _simulated_annealing(centers, rng, time_budget):
    """
    Simulated annealing using KDTree for fast radius evaluation.
    Only rebuilds KDTree periodically; uses incremental updates otherwise.
    """
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_r = _compute_radius(current)

    T_start = 0.012
    T_end = 0.00003
    # Estimate steps: each step ~0.0002s with KDTree
    est_steps = int(time_budget / 0.0002)
    n_steps = max(20000, min(150000, est_steps))
    cooling = (T_end / T_start) ** (1.0 / max(n_steps, 1))
    T = T_start

    # Build initial KDTree
    tree = cKDTree(current)
    min_pair = _fast_min_pair_dist(current)
    wall_x = 0.5 - np.abs(current[:, 0])
    wall_y = 0.5 - np.abs(current[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    curr_r = min(min_pair, min_wall)

    # Track which point was last moved for incremental updates
    last_moved = None

    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break

        # Move generation
        i = rng.integers(0, N)
        move_size = 0.002 + 0.008 * (T / T_start)

        if rng.random() < 0.05:
            move_size *= 3.0

        direction = rng.normal(size=2)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        else:
            direction = np.array([1.0, 0.0])

        old_pos = current[i].copy()
        trial = current.copy()
        trial[i] += direction * move_size
        trial[i] = np.clip(trial[i], -0.5, 0.5)

        # Compute trial radius using KDTree
        # For speed, only check the moved point's neighborhood
        trial_tree = cKDTree(trial)
        # Get nearest neighbor distance for moved point
        d_nn, _ = trial_tree.query(trial[i], k=2)
        trial_min_pair_local = d_nn[1] / 2.0

        # Also need global min pair - use KDTree
        # For efficiency, sample a few random points
        sample_idx = rng.integers(0, N, size=min(10, N))
        for si in sample_idx:
            d_s, _ = trial_tree.query(trial[si], k=2)
            if d_s[1] / 2.0 < trial_min_pair_local:
                trial_min_pair_local = d_s[1] / 2.0

        # Wall distances for trial
        trial_wall_x = 0.5 - np.abs(trial[:, 0])
        trial_wall_y = 0.5 - np.abs(trial[:, 1])
        trial_min_wall = np.min(np.minimum(trial_wall_x, trial_wall_y))

        trial_r = min(trial_min_pair_local, trial_min_wall)

        delta_r = trial_r - curr_r

        if delta_r >= 0:
            current = trial
            tree = trial_tree
            last_moved = i
            if trial_r > best_r:
                best_r = trial_r
                best = trial.copy()
        else:
            if rng.random() < np.exp(delta_r / max(T, 1e-10)):
                current = trial
                tree = trial_tree
                last_moved = i

        # Update current radius tracking
        if last_moved is not None:
            # Recompute min_pair more accurately periodically
            if step % 50 == 0:
                min_pair = _fast_min_pair_dist(current)
            else:
                # Quick estimate: check moved point's neighbors
                d_nn, _ = tree.query(current[last_moved], k=2)
                local_min = d_nn[1] / 2.0
                # Also check a few random points
                for _ in range(3):
                    ri = rng.integers(0, N)
                    d_r, _ = tree.query(current[ri], k=2)
                    if d_r[1] / 2.0 < local_min:
                        local_min = d_r[1] / 2.0
                min_pair = local_min

            wall_x = 0.5 - np.abs(current[:, 0])
            wall_y = 0.5 - np.abs(current[:, 1])
            min_wall = np.min(np.minimum(wall_x, wall_y))
            curr_r = min(min_pair, min_wall)

        T *= cooling

    # Final accurate radius computation
    final_r = _compute_radius(current)
    if final_r > best_r:
        best_r = final_r
        best = current.copy()

    return best, best_r


def _lbfgs_soft_min(x0, beta, time_budget):
    """Run L-BFGS-B to minimize the negative soft-min objective."""
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_min_obj(x, beta)
        g = _soft_min_grad(x, beta)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxfun': 15000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    """Run L-BFGS-B to minimize violation for a specific target radius."""
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_budget:
            return 0.0, np.zeros_like(x)
        f = _soft_violation_obj(x, r_target, beta)
        g = _soft_violation_grad(x, r_target, beta)
        return f, g

    try:
        result = opt.minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13,
                     'maxfun': 15000, 'disp': False}
        )
        polished = result.x.reshape(-1, 2)
        polished = np.clip(polished, -0.5, 0.5)
        return polished
    except Exception:
        return x0.reshape(-1, 2)


def _soft_min_progressive(centers, time_budget):
    """Progressive sharpening: increase beta for tighter min approximation."""
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    betas = [80, 150, 300, 600, 1200, 2500, 5000]

    for beta in betas:
        if time.time() - start_time > time_budget:
            break

        x0 = best.flatten()

        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        lbfgs_budget = min(12.0, remaining * 0.85)

        trial = _lbfgs_soft_min(x0, beta, lbfgs_budget)
        trial_r = _compute_radius(trial)

        if trial_r > best_r:
            best_r = trial_r
            best = trial

    return best, best_r


def _high_precision_polish(centers, time_budget):
    """Final high-precision polish."""
    start_time = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)

    # Phase 1: Soft-min with high beta
    if time.time() - start_time < time_budget - 10:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 10:
            best, best_r = _soft_min_progressive(best, min(25.0, remaining - 5))

    # Phase 2: Feasibility-driven push
    if time.time() - start_time < time_budget - 5:
        for push_frac in [1.001, 1.003, 1.006, 1.01, 1.015, 1.02]:
            if time.time() - start_time > time_budget - 3:
                break
            r_push = best_r * push_frac
            if r_push > 0.08:
                break
            remaining = time_budget - (time.time() - start_time)
            if remaining < 3:
                break
            lbfgs_budget = min(8.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), r_push, 3000.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial

    # Phase 3: Final very high beta soft-min
    if time.time() - start_time < time_budget - 3:
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            lbfgs_budget = min(6.0, remaining * 0.85)
            trial = _lbfgs_soft_min(best.flatten(), 8000.0, lbfgs_budget)
            trial_r = _compute_radius(trial)
            if trial_r > best_r:
                best_r = trial_r
                best = trial

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = np.random.default_rng(777)

    # 1. Initialization
    inits = [
        _initial_grid(),
        _hex_init(),
    ]

    rng_init = np.random.default_rng(999)
    random_init = rng_init.uniform(-0.45, 0.45, (N, 2))
    inits.append(random_init)

    best_centers = None
    best_radius = 0

    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r
            best_centers = init.copy()

    if best_centers is None:
        best_centers = _initial_grid()
        best_radius = _compute_radius(best_centers)

    # 2. Simulated Annealing - more time budget, multiple restarts
    sa_budget_total = min(65.0, (deadline - time.time()) * 0.50)

    if sa_budget_total > 15:
        # First SA run - longer
        sa1_budget = min(35.0, sa_budget_total * 0.55)
        best_centers, best_radius = _simulated_annealing(best_centers, rng, sa1_budget)

        # Second SA run with perturbation
        if time.time() < deadline - 40:
            rng2 = np.random.default_rng(1234)
            perturbed = best_centers + rng2.normal(0, 0.015, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            sa2_budget = min(25.0, (deadline - time.time()) - 45)
            if sa2_budget > 8:
                c2, r2 = _simulated_annealing(perturbed, rng2, sa2_budget)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = c2

        # Third SA run if time permits
        if time.time() < deadline - 35:
            rng3 = np.random.default_rng(5678)
            perturbed3 = best_centers + rng3.normal(0, 0.01, (N, 2))
            perturbed3 = np.clip(perturbed3, -0.49, 0.49)
            sa3_budget = min(20.0, (deadline - time.time()) - 40)
            if sa3_budget > 5:
                c3, r3 = _simulated_annealing(perturbed3, rng3, sa3_budget)
                if r3 > best_radius:
                    best_radius = r3
                    best_centers = c3

    # 3. Soft-min progressive sharpening
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        best_centers, best_radius = _soft_min_progressive(
            best_centers, time_budget=min(35.0, remaining - 10)
        )

    # 4. High Precision Polish
    if time.time() < deadline - 8:
        best_centers, best_radius = _high_precision_polish(
            best_centers, time_budget=min(20.0, deadline - time.time() - 3)
        )

    # Final Validation
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()

    return best_centers
