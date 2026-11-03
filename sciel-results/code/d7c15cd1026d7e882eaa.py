import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill


def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if np.any(points > 0.5) or np.any(points < -0.5):
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
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
    """High-precision polish using coordinate descent on the actual min-distance objective."""
    start = time.time()
    pts = points.copy()
    best_score = _score(pts)
    best_pts = pts.copy()

    lr = 0.01
    decay = 0.99

    while time.time() - start < max_time and lr > 1e-6:
        # Find bottleneck
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists) / 2.0

        wall_dists = 0.5 - np.abs(pts)
        min_wall_val = np.min(wall_dists)

        improved = False

        if min_pair_val < min_wall_val:
            idx = np.unravel_index(np.argmin(dists), dists.shape)
            i, j = idx
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                d /= dist
                pts[i] += lr * d
                pts[j] -= lr * d
                improved = True
        else:
            flat_idx = np.argmin(wall_dists)
            i, k = flat_idx // 3, flat_idx % 3
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            pts[i, k] += lr * s
            improved = True

        pts = np.clip(pts, -0.5, 0.5)

        if improved:
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

        lr *= decay

    return best_pts, best_score


def construct():
    start_time = time.time()
    best_points = None
    best_score = 0.0

    # Multi-restart with dynamics shaking and L-BFGS-B optimization
    num_restarts = 0
    while time.time() - start_time < DEADLINE - 20:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 15:
            break

        # Generate diverse initial configurations
        if num_restarts % 3 == 0:
            points = _init_grid(N)
        elif num_restarts % 3 == 1:
            points = _init_bcc(N)
        else:
            points = _init_random(N, seed=num_restarts * 12345 + 678)

        # Dynamics shaking to escape local minima
        shake_time = min(3.0, max(1.0, remaining * 0.2))
        points = _dynamics_shake(points, time_budget=shake_time)

        # Optimize with L-BFGS-B
        opt_time = min(8.0, max(2.0, remaining * 0.5))
        points, sc = _optimize(points, max_time=opt_time)

        # Final polish
        polish_time = min(5.0, max(1.0, (DEADLINE - 5) - (time.time() - start_time)))
        if polish_time > 0.5:
            points, sc = _polish(points, polish_time)
        else:
            sc = _score(points)

        if sc > best_score:
            best_score = sc
            best_points = points.copy()

        num_restarts += 1

    if best_points is None:
        best_points = _init_grid(N)

    return best_points


if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
