"""
Combined sphere packing: 50 equal spheres in unit cube.

Strategy:
1. Multiple diverse initializations (FCC, BCC, grid, random).
2. Smooth surrogate optimization with aggressive beta annealing using logsumexp.
3. Vectorized gradient computation for speed.
4. Active-set refinement and final high-precision polish.
5. Deadline guarding to stay within time limits.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

N = 50
CUBE_HALF = 0.5
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# Precompute triangle indices once
TRI_I, TRI_J = np.triu_indices(N, k=1)
N_PAIRS = len(TRI_I)


def _get_radius(c):
    """Compute the exact maximum common radius for centers c (N,3)."""
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
    min_wall_dist = np.min(dist_to_walls)
    return float(min(min_pair_dist / 2.0, min_wall_dist))


def _smooth_objective_and_grad(c_flat, beta):
    """
    Vectorized soft-min surrogate for the packing objective.
    Returns (value, gradient) for minimization of -soft_min(g_k).
    """
    c = c_flat.reshape(N, 3)

    if np.any(np.abs(c) > 0.5 + 1e-9):
        return 1e6, np.zeros_like(c_flat)

    # Pairwise distances
    diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d_full = np.sqrt(np.sum(diff_full ** 2, axis=2))
    d_pairs = d_full[TRI_I, TRI_J] / 2.0

    # Wall distances (6 per sphere)
    d_x_pos = 0.5 - c[:, 0]
    d_x_neg = c[:, 0] + 0.5
    d_y_pos = 0.5 - c[:, 1]
    d_y_neg = c[:, 1] + 0.5
    d_z_pos = 0.5 - c[:, 2]
    d_z_neg = c[:, 2] + 0.5

    wall_dists = np.concatenate([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg])
    g_all = np.concatenate([d_pairs, wall_dists])

    # Soft min value using logsumexp for numerical stability
    neg_beta_g = -beta * g_all
    lse = logsumexp(neg_beta_g)
    val = -lse / beta

    # Weights
    weights = np.exp(neg_beta_g - lse)

    # Gradient: vectorized pairwise contribution
    grad_c = np.zeros_like(c)

    diff_ij = c[TRI_I] - c[TRI_J]
    d_ij = d_full[TRI_I, TRI_J]
    safe_d = np.where(d_ij > 1e-14, d_ij, 1.0)
    unit_vecs = diff_ij / safe_d[:, np.newaxis]
    w_pair = weights[:N_PAIRS] * 0.5 / safe_d
    contrib = w_pair[:, np.newaxis] * unit_vecs

    np.add.at(grad_c, TRI_I, contrib)
    np.add.at(grad_c, TRI_J, -contrib)

    # Wall gradients
    ws = N_PAIRS
    grad_c[:, 0] -= weights[ws:ws + N]
    grad_c[:, 0] += weights[ws + N:ws + 2 * N]
    grad_c[:, 1] -= weights[ws + 2 * N:ws + 3 * N]
    grad_c[:, 1] += weights[ws + 3 * N:ws + 4 * N]
    grad_c[:, 2] -= weights[ws + 4 * N:ws + 5 * N]
    grad_c[:, 2] += weights[ws + 5 * N:ws + 6 * N]

    return -val, -grad_c.flatten()


def _run_smooth_optimization(c_init, beta, max_iter, bounds_range=0.499):
    """Run L-BFGS-B on the smooth surrogate."""
    x0 = c_init.flatten()
    bounds = [(-bounds_range, bounds_range)] * (N * 3)

    def fun(x):
        val, _ = _smooth_objective_and_grad(x, beta)
        return val

    def jac(x):
        _, grad = _smooth_objective_and_grad(x, beta)
        return grad

    try:
        result = minimize(
            fun, x0, jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                     'maxls': 100, 'maxfun': max_iter * 10, 'maxcor': 50}
        )
        return np.clip(result.x.reshape(N, 3), -bounds_range, bounds_range)
    except Exception:
        return c_init.copy()


def _run_direct_optimization(c_init, max_iter=200):
    """Run L-BFGS-B on the exact non-smooth objective."""
    x0 = c_init.flatten()
    bounds = [(-0.499, 0.499)] * (N * 3)

    def fun(x):
        return -_get_radius(x.reshape(N, 3))

    try:
        result = minimize(
            fun, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        return np.clip(result.x.reshape(N, 3), -0.499, 0.499)
    except Exception:
        return c_init.copy()


def _generate_fcc_lattice(n_points):
    """
    Generate points on an FCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 4
    points = []

    # FCC: corners + face centers
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])

    # Face centers
    for i in range(n_div):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([(i + 0.5) / n_div, j / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div):
            for k in range(n_div + 1):
                points.append([i / n_div, (j + 0.5) / n_div, k / n_div])
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div):
                points.append([i / n_div, j / n_div, (k + 0.5) / n_div])

    points = np.array(points)
    points = points - 0.5

    # Keep only points within the cube
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]

    rng = np.random.default_rng(456)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]

    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.09:
            selected.append(p)

    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)

    return np.array(selected[:n_points])


def _generate_bcc_lattice(n_points):
    """
    Generate points on a BCC lattice scaled to fit in [-0.5, 0.5]^3.
    """
    n_div = 5
    points = []

    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                points.append([i / n_div, j / n_div, k / n_div])

    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])

    points = np.array(points)
    points = points - 0.5

    rng = np.random.default_rng(123)
    indices = np.arange(len(points))
    rng.shuffle(indices)
    points_shuffled = points[indices]

    selected = [points_shuffled[0]]
    for i in range(1, len(points_shuffled)):
        if len(selected) >= n_points:
            break
        p = points_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.08:
            selected.append(p)

    while len(selected) < n_points:
        p = rng.uniform(-0.45, 0.45, size=3)
        selected.append(p)

    return np.array(selected[:n_points])


def _active_set_refinement(c_init, r_target, max_iter=100):
    """
    Active-set refinement: identify constraints active at r_target and
    optimize only over those degrees of freedom.
    """
    c = c_init.copy()
    r = _get_radius(c)

    # Identify active constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)

    dist_to_walls = np.min(0.5 - np.abs(c), axis=1)
    min_wall_dist = np.min(dist_to_walls)

    # If the bottleneck is pair distance, try to increase it
    if min_pair_dist / 2.0 < min_wall_dist:
        pair_dists = d[TRI_I, TRI_J]
        close_pairs = np.where(pair_dists < min_pair_dist * 1.01)[0]
        if len(close_pairs) > 0:
            involved = np.unique(np.concatenate([TRI_I[close_pairs], TRI_J[close_pairs]]))
            rng = np.random.RandomState(42)
            perturb = rng.normal(0, 0.005, size=(len(involved), 3))
            c[involved] += perturb
            c = np.clip(c, -0.499, 0.499)
            r_new = _get_radius(c)
            if r_new > r:
                c, r = c, r_new

    # Final direct optimization
    c_polished = _run_direct_optimization(c, max_iter)
    r_polished = _get_radius(c_polished)
    if r_polished > r:
        c, r = c_polished, r_polished

    return c, r


def _annealed_optimization(c_init, start_time, deadline):
    """
    Multi-stage annealing with aggressive beta schedule and time-proportional
    iteration allocation.
    """
    betas = [10, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 100000, 200000, 400000]

    c_best = c_init.copy()
    r_best = _get_radius(c_init)

    for idx, beta in enumerate(betas):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3.0:
            break

        n_remaining = len(betas) - idx
        stage_time = max(1.0, remaining / n_remaining * 0.85)
        max_iter = int(min(200, max(20, stage_time * 15 / (1 + 0.1 * np.log2(beta)))))

        c_new = _run_smooth_optimization(c_best, beta, max_iter)
        r_new = _get_radius(c_new)
        if r_new > r_best:
            r_best = r_new
            c_best = c_new.copy()

    # Active-set refinement
    elapsed = time.time() - start_time
    if deadline - elapsed > 5.0:
        c_refined, r_refined = _active_set_refinement(c_best, r_best)
        if r_refined > r_best:
            c_best = c_refined.copy()
            r_best = r_refined

    # Final direct polish
    elapsed = time.time() - start_time
    if deadline - elapsed > 3.0:
        c_polished = _run_direct_optimization(c_best, max_iter=150)
        r_polished = _get_radius(c_polished)
        if r_polished > r_best:
            c_best = c_polished.copy()
            r_best = r_polished

    return c_best, r_best


def _local_polish(c, deadline):
    """
    Additional local polish using very high beta values and targeted coordinate-wise refinement.
    """
    best_c = c.copy()
    best_val = _get_radius(best_c)

    if time.time() > deadline - 5:
        return best_c, best_val

    # Try a few more optimization passes with different betas
    for beta in [20000, 50000, 100000, 200000, 500000]:
        if time.time() > deadline - 3:
            break

        c_new = _run_smooth_optimization(best_c, beta, 400, bounds_range=0.499999)
        new_val = _get_radius(c_new)

        if new_val > best_val:
            best_val = new_val
            best_c = c_new.copy()

    # Targeted coordinate-wise refinement
    if time.time() < deadline - 1:
        wall = 0.5 - np.abs(best_c)
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d) / 2.0
        min_wall_val = np.min(wall)
        r_current = min(min_pair_dist, min_wall_val)

        active_spheres = set()
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] / 2.0 < r_current * 1.01:
                    active_spheres.add(i)
                    active_spheres.add(j)
        for i in range(N):
            for k in range(3):
                if wall[i, k] < r_current * 1.01:
                    active_spheres.add(i)

        if not active_spheres:
            active_spheres = set(range(N))

        step_sizes = [1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
        for step in step_sizes:
            if time.time() > deadline - 0.5:
                break
            improved = True
            while improved and time.time() < deadline - 0.5:
                improved = False
                for i in active_spheres:
                    for k in range(3):
                        for sign in [-1, 1]:
                            c_trial = best_c.copy()
                            c_trial[i, k] += sign * step
                            c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                            val_trial = _get_radius(c_trial)
                            if val_trial > best_val + 1e-12:
                                best_val = val_trial
                                best_c = c_trial
                                improved = True

    # Final active-set refinement
    if time.time() < deadline - 0.3:
        wall = 0.5 - np.abs(best_c)
        d = cdist(best_c, best_c)
        np.fill_diagonal(d, np.inf)
        min_pair_val = np.min(d) / 2.0
        min_wall_val = np.min(wall)
        r_current = min(min_pair_val, min_wall_val)

        tight_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] / 2.0 < r_current * 1.001:
                    tight_pairs.append((i, j))

        tight_walls = []
        for i in range(N):
            for k in range(3):
                if wall[i, k] < r_current * 1.001:
                    tight_walls.append((i, k))

        for i, j in tight_pairs[:10]:
            if time.time() > deadline - 0.1:
                break
            diff = best_c[i] - best_c[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                for delta in [1e-7, 5e-8, 1e-8]:
                    c_trial = best_c.copy()
                    c_trial[i] += direction * delta
                    c_trial[j] -= direction * delta
                    c_trial = np.clip(c_trial, -0.4999999, 0.4999999)
                    val_trial = _get_radius(c_trial)
                    if val_trial > best_val + 1e-13:
                        best_val = val_trial
                        best_c = c_trial
                        break

        for i, k in tight_walls[:10]:
            if time.time() > deadline - 0.1:
                break
            if best_c[i, k] > 0:
                direction = np.zeros(3)
                direction[k] = -1
            else:
                direction = np.zeros(3)
                direction[k] = 1

            for delta in [1e-7, 5e-8, 1e-8]:
                c_trial = best_c.copy()
                c_trial[i] += direction * delta
                c_trial = np.clip(c_trial, -0.4999999, 0.4999999)
                val_trial = _get_radius(c_trial)
                if val_trial > best_val + 1e-13:
                    best_val = val_trial
                    best_c = c_trial
                    break

    return best_c, best_val


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    rng = np.random.default_rng(42)

    # Generate multiple initial configurations
    c_fcc = _generate_fcc_lattice(N)
    c_bcc = _generate_bcc_lattice(N)

    initial_configs = [c_fcc.copy(), c_bcc.copy()]

    # Create variants with different perturbations
    for seed in [1, 2, 3, 4]:
        c_variant = c_fcc.copy()
        noise = rng.normal(0, 0.015, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)

    # BCC variants
    for seed in [5, 6]:
        c_variant = c_bcc.copy()
        noise = rng.normal(0, 0.012, size=(N, 3))
        c_variant = c_variant + noise
        c_variant = np.clip(c_variant, -0.45, 0.45)
        initial_configs.append(c_variant)

    # Grid-based initialization
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = np.arange(len(grid))
    rng.shuffle(indices)
    grid_shuffled = grid[indices]

    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)

    while len(selected) < N:
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
            p = rng.uniform(-0.45, 0.45, size=3)
            selected.append(p)

    initial_configs.append(np.array(selected[:N]))

    # Optimize from each initial configuration
    best_c = initial_configs[0].copy()
    best_val = _get_radius(best_c)

    for idx, c_init in enumerate(initial_configs):
        elapsed = time.time() - start_time
        if deadline - elapsed < 40:
            break

        try:
            c_opt, val = _annealed_optimization(c_init, start_time, deadline)
            if val > best_val:
                best_val = val
                best_c = c_opt.copy()
        except Exception:
            pass

    # Final polish on the best result
    elapsed = time.time() - start_time
    if deadline - elapsed > 8:
        best_c, best_val = _local_polish(best_c, deadline)

    # Ensure valid output
    best_c = np.clip(best_c, -0.499999999, 0.499999999)
    best_c = np.asarray(best_c, dtype=float)

    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    # Final validation: ensure no coincident centers
    d = cdist(best_c, best_c)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-10:
        rng_fix = np.random.default_rng(999)
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(best_c[i] - best_c[j]) < 1e-10:
                    best_c[j] += rng_fix.uniform(-1e-8, 1e-8, size=3)
        best_c = np.clip(best_c, -0.499999999, 0.499999999)

    return best_c
