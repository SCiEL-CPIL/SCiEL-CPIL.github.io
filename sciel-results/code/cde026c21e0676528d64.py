import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE_SECONDS = 115  # Safety margin before 120s limit


def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0

    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0

    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)

    return min(min_pair, min_wall)


def _soft_min(values, beta):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    v_min = v.min()
    exp_terms = np.exp(-beta * (v - v_min))
    log_sum = np.log(np.sum(exp_terms))
    return v_min - log_sum / beta


def _objective_flat(x, beta=50.0):
    """
    Objective function for optimization: negative of a smooth approximation to the minimum radius.
    """
    points = x.reshape(N, 3)

    # Wall distances
    wall_dists = 0.5 - np.abs(points)

    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)

    # Collect all constraint values
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx] / 2.0
    wall_vals = wall_dists.flatten()

    all_vals = np.concatenate([pair_dists, wall_vals])
    smooth_min = _soft_min(all_vals, beta)
    return -smooth_min


def _init_grid(n):
    """Generate an initial grid-based configuration."""
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                points.append([
                    -0.5 + (i + 0.5) * (1.0 / 4.0),
                    -0.5 + (j + 0.5) * (1.0 / 4.0),
                    -0.5 + (k + 0.5) * (1.0 / 4.0)
                ])

    points = np.array(points)

    # Randomly select 50
    idx = np.random.choice(len(points), n, replace=False)
    points = points[idx]

    # Add small random jitter to break symmetry
    points += np.random.randn(n, 3) * 0.01
    points = np.clip(points, -0.49, 0.49)

    return points


def _init_random(n, seed=None):
    """Generate a random initial configuration."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random
    return rng.uniform(-0.45, 0.45, (n, 3))


def _init_bcc(n):
    """Body-centered cubic lattice initialization, scaled to fit N points."""
    # BCC has 2 points per conventional cell
    # For N=50, we need 25 conventional cells
    n_cells = N // 2
    side = int(np.ceil(n_cells ** (1.0 / 3.0)))
    points = []
    for iz in range(side):
        for iy in range(side):
            for ix in range(side):
                if len(points) >= N:
                    break
                points.append([ix, iy, iz])
                if len(points) < N:
                    points.append([ix + 0.5, iy + 0.5, iz + 0.5])

    pts = np.array(points[:N], dtype=float)
    max_coord = pts.max()
    if max_coord > 0:
        scale = (HALF - 0.01) / max_coord
        pts = pts * scale - scale * max_coord / 2.0
    return pts


def _dynamics_shake(points, time_budget=5.0):
    """
    Rigid-body dynamics simulation with repulsive forces and damping.
    Uses velocity-Verlet-like integration with strong damping to shake out local minima.
    """
    start_time = time.time()
    positions = points.copy().astype(float)
    velocities = np.zeros_like(positions)

    # Force parameters
    repulsion_strength = 0.5
    damping = 0.9  # Strong damping
    dt = 0.005
    min_dist_cutoff = 0.01  # Avoid division by zero
    max_force = 100.0

    def _compute_forces(pos):
        """Compute repulsive forces between all pairs and wall repulsion."""
        forces = np.zeros_like(pos)

        # Pairwise repulsive forces: F = strength / d^2 * direction
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1e10)
        dists = np.sqrt(dists_sq)

        # Force magnitude: repulsion_strength / d^2, capped
        force_mags = repulsion_strength / np.maximum(dists_sq, min_dist_cutoff ** 2)
        force_mags = np.minimum(force_mags, max_force)
        np.fill_diagonal(force_mags, 0.0)

        # Force direction: unit vector from j to i
        # F_ij = force_mag * (pos_i - pos_j) / dist
        force_dirs = diff / np.maximum(dists[:, :, np.newaxis], min_dist_cutoff)
        forces += np.sum(force_mags[:, :, np.newaxis] * force_dirs, axis=1)

        # Wall repulsion: push away from walls
        wall_dists = HALF - np.abs(pos)
        # If close to wall, push inward
        for axis in range(3):
            # Positive wall
            close_pos = pos[:, axis] > 0
            wall_dist_pos = HALF - pos[close_pos, axis]
            force_wall_pos = np.zeros_like(wall_dist_pos)
            close_to_wall = wall_dist_pos < 0.1
            force_wall_pos[close_to_wall] = repulsion_strength / np.maximum(wall_dist_pos[close_to_wall] ** 2, min_dist_cutoff ** 2)
            force_wall_pos = np.minimum(force_wall_pos, max_force)
            forces[close_pos, axis] -= force_wall_pos

            # Negative wall
            close_neg = pos[:, axis] < 0
            wall_dist_neg = HALF + pos[close_neg, axis]
            force_wall_neg = np.zeros_like(wall_dist_neg)
            close_to_wall = wall_dist_neg < 0.1
            force_wall_neg[close_to_wall] = repulsion_strength / np.maximum(wall_dist_neg[close_to_wall] ** 2, min_dist_cutoff ** 2)
            force_wall_neg = np.minimum(force_wall_neg, max_force)
            forces[close_neg, axis] += force_wall_neg

        return forces

    # Main simulation loop
    while time.time() - start_time < time_budget:
        forces = _compute_forces(positions)

        # Velocity-Verlet-like update with damping
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt

        # Keep within bounds
        positions = np.clip(positions, -HALF + 1e-10, HALF - 1e-10)

        # If system has settled (small velocities), break early
        if np.max(np.abs(velocities)) < 1e-6:
            break

    return positions


def _optimize(points, max_time, beta_start=50.0, beta_end=200.0):
    """
    Optimize the configuration using L-BFGS-B on the smooth objective with beta annealing.
    """
    start_time = time.time()
    x0 = points.flatten()

    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)

    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 3))

    # Beta annealing schedule
    n_stages = 4
    betas = np.linspace(beta_start, beta_end, n_stages)

    for beta in betas:
        if time.time() - start_time > max_time:
            break

        try:
            res = minimize(
                _objective_flat,
                best_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            if res.fun < _objective_flat(best_x, beta):
                candidate = res.x
                candidate_score = _score(candidate.reshape(N, 3))
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = candidate

        except Exception:
            pass

    return best_x.reshape(N, 3), best_score


def _polish(points, max_time=5.0):
    """High-precision polish with very tight tolerances."""
    start_time = time.time()
    x0 = points.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    best_x = x0.copy()
    best_score = _score(x0.reshape(N, 3))

    # Use high beta for precision
    beta_high = 500.0

    try:
        res = minimize(
            _objective_flat,
            best_x,
            args=(beta_high,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
        )

        candidate = res.x
        candidate_score = _score(candidate.reshape(N, 3))
        if candidate_score > best_score:
            best_score = candidate_score
            best_x = candidate

    except Exception:
        pass

    # Final exact polish
    if time.time() - start_time < max_time - 1.0:
        def _exact_neg_radius(x):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(d, np.inf)
            pair_min = np.min(d) / 2.0
            wall = HALF - np.abs(c)
            wall_min = np.min(wall)
            return -min(pair_min, wall_min)

        try:
            res2 = minimize(
                _exact_neg_radius,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )

            candidate = res2.x
            candidate_score = _score(candidate.reshape(N, 3))
            if candidate_score > best_score:
                best_score = candidate_score
                best_x = candidate

        except Exception:
            pass

    return best_x.reshape(N, 3), best_score


def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    best_points = None
    best_score = 0.0

    # Generate diverse initializations
    inits = []

    # Grid init
    grid_init = _init_grid(N)
    inits.append(grid_init)

    # BCC init
    bcc_init = _init_bcc(N)
    inits.append(bcc_init)

    # Random inits with different seeds
    for seed in [0, 42, 123, 456]:
        inits.append(_init_random(N, seed=seed))

    # Process each initialization
    for init_points in inits:
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break

        # Optional: dynamics shake to escape local minima
        if time.time() - start_time < DEADLINE_SECONDS - 15:
            shaken = _dynamics_shake(init_points, time_budget=3.0)
            shaken_score = _score(shaken)
            if shaken_score > _score(init_points):
                init_points = shaken

        # Optimize
        opt_points, opt_score = _optimize(init_points, max_time=15.0)

        # Polish
        if time.time() - start_time < DEADLINE_SECONDS - 5:
            polished_points, polished_score = _polish(opt_points, max_time=5.0)
            if polished_score > opt_score:
                opt_points = polished_points
                opt_score = polished_score

        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()

        # If we have a good score, try refining further
        if best_score > 0.13:
            for _ in range(2):
                if time.time() - start_time > DEADLINE_SECONDS - 5:
                    break
                refined_points, refined_score = _optimize(best_points, max_time=5.0)
                if time.time() - start_time < DEADLINE_SECONDS - 3:
                    refined_points, refined_score = _polish(refined_points, max_time=3.0)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_points.copy()

    if best_points is None:
        best_points = _init_grid(N)

    # Final safety clip
    best_points = np.clip(best_points, -HALF + 1e-10, HALF - 1e-10)

    return best_points


if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
