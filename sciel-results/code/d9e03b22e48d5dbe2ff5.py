"""
Pack 200 equal circles in a unit square.

Strategy:
1. Multi-start: generate several initial configurations (hex lattices with
   slightly different spacing/phase, plus a jittered variant).
2. For each start, run a force-directed relaxation to remove overlaps.
3. Then run a smooth soft-min objective with L-BFGS-B, increasing the
   sharpness parameter beta to sharpen the min.
4. Use a projected-gradient-descent / local-search phase to escape local
   minima, with adaptive step size and random perturbations.
5. Final high-precision polish with very high beta.
6. Track the best radius found so far and return the best configuration.

We keep the public function `construct()` returning an (N, 2) array.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 200
_TIME_LIMIT = 149  # safety margin before the 150s kill

def _get_radius(pts):
    """
    Compute the maximum common radius for a set of centers:
    min of (distance to walls) and (half of minimum pairwise distance).
    """
    if len(pts) == 0:
        return 0.0

    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)

    # Use KDTree for faster nearest neighbor search
    tree = cKDTree(pts)
    min_pair, _ = tree.query(pts, k=2)
    min_pair = np.min(min_pair) / 2.0

    return min(min_wall, min_pair)

def _soft_min(vals, beta):
    """
    Soft minimum: -1/beta * log(sum(exp(-beta * x_i))).
    Numerically stable via max shift.
    """
    y = -beta * vals
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return np.min(vals)
    shifted = y - max_y
    sum_exp = np.sum(np.exp(shifted))
    if sum_exp == 0:
        return np.min(vals)
    return -(max_y + np.log(sum_exp)) / beta

def _objective_and_grad(x, beta):
    """
    Objective for minimization: -soft_min(constraints).
    Constraints:
      - 4 wall distances per point
      - pairwise distances / 2
    Returns (value, gradient).
    """
    pts = x.reshape((N, 2))

    # Wall values
    C_wall = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Use KDTree to find only close pairs (within a reasonable distance)
    tree = cKDTree(pts)
    # Only consider pairs within a certain distance to reduce computation
    search_radius = 0.15
    close_pairs = tree.query_pairs(search_radius)
    
    if len(close_pairs) == 0:
        # No close pairs, just wall constraints
        C = C_wall.copy()
        y = -beta * C
        max_y = np.max(y)
        if not np.isfinite(max_y):
            min_val = np.min(C)
            return -min_val, np.zeros_like(x)
        shifted = y - max_y
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        if sum_exp == 0:
            min_val = np.min(C)
            return -min_val, np.zeros_like(x)
        soft_min_val = -(max_y + np.log(sum_exp)) / beta
        obj_val = -soft_min_val
        weights = exp_shifted / sum_exp
        grad_pts = np.zeros((N, 2))
        grad_x_left = weights[0:N] * 1.0
        grad_x_right = weights[N:2*N] * (-1.0)
        grad_y_bottom = weights[2*N:3*N] * 1.0
        grad_y_top = weights[3*N:4*N] * (-1.0)
        grad_pts[:, 0] += grad_x_left + grad_x_right
        grad_pts[:, 1] += grad_y_bottom + grad_y_top
        grad_pts = -grad_pts / beta
        return obj_val, grad_pts.flatten()

    i_idx = np.array([p[0] for p in close_pairs])
    j_idx = np.array([p[1] for p in close_pairs])
    
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C_pair = dists / 2.0

    # Combine all constraint values
    C = np.concatenate([C_wall, C_pair])

    # Soft min
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        min_val = np.min(C)
        return -min_val, np.zeros_like(x)

    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val

    # Weights
    weights = exp_shifted / sum_exp

    # Gradient
    grad_pts = np.zeros((N, 2))

    # Wall gradients
    grad_x_left = weights[0:N] * 1.0
    grad_x_right = weights[N:2*N] * (-1.0)
    grad_y_bottom = weights[2*N:3*N] * 1.0
    grad_y_top = weights[3*N:4*N] * (-1.0)

    grad_pts[:, 0] += grad_x_left + grad_x_right
    grad_pts[:, 1] += grad_y_bottom + grad_y_top

    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)

    grad_pair_x_i = dx_safe / (2.0 * dists_safe)
    grad_pair_x_j = -dx_safe / (2.0 * dists_safe)
    grad_pair_y_i = dy_safe / (2.0 * dists_safe)
    grad_pair_y_j = -dy_safe / (2.0 * dists_safe)

    grad_pair_x_i = np.where(valid, grad_pair_x_i, 0.0)
    grad_pair_x_j = np.where(valid, grad_pair_x_j, 0.0)
    grad_pair_y_i = np.where(valid, grad_pair_y_i, 0.0)
    grad_pair_y_j = np.where(valid, grad_pair_y_j, 0.0)

    pair_weights = weights[4*N:]

    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)

    grad_pts = -grad_pts / beta

    return obj_val, grad_pts.flatten()

def _generate_hex_initial(spacing, phase_shift=0.0, margin=0.001):
    """
    Generate a hexagonal lattice packing with given spacing and phase shift.
    Returns N points inside [-0.5, 0.5]^2.
    """
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + row_h / 2 + phase_shift
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + spacing / 2 + shift + (phase_shift if row_idx % 2 == 0 else 0.0)
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1

    pts = np.array(points)
    mask = (pts[:, 0] >= -0.5 + margin) & (pts[:, 0] <= 0.5 - margin) & \
           (pts[:, 1] >= -0.5 + margin) & (pts[:, 1] <= 0.5 - margin)
    pts = pts[mask]

    if len(pts) < N:
        # Fallback: random
        return np.random.uniform(-0.4, 0.4, (N, 2))

    # Select N points closest to the center
    dists_to_center = np.sum(pts**2, axis=1)
    indices = np.argsort(dists_to_center)
    pts = pts[indices[:N]]

    return pts

def _generate_jittered_hex(spacing, jitter=0.002, seed=0):
    """
    Generate hex lattice with small random jitter to break symmetry.
    """
    rng = np.random.RandomState(seed)
    pts = _generate_hex_initial(spacing, phase_shift=0.0)
    pts = pts + rng.uniform(-jitter, jitter, size=pts.shape)
    pts = np.clip(pts, -0.499, 0.499)
    return pts

def _force_directed_phase(pts, time_limit, iterations=None):
    """
    Force-directed repulsion to spread out overlapping circles.
    Uses KDTree for efficient neighbor finding and boundary repulsion.
    """
    if iterations is None:
        iterations = max(300, int(time_limit * 100))

    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    step = 0.004
    lr_decay = 0.998

    for it in range(iterations):
        if time.time() - t_start > time_limit:
            break

        forces = np.zeros_like(pts)

        # Use KDTree to find close pairs
        tree = cKDTree(pts)
        threshold = 0.12
        close_pairs = tree.query_pairs(threshold)

        for i, j in close_pairs:
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d2 = dx*dx + dy*dy
            if d2 > 1e-12:
                d = np.sqrt(d2)
                f = 1.0 / (d2 * d)  # 1/d^3
                forces[i, 0] += f * dx
                forces[i, 1] += f * dy
                forces[j, 0] -= f * dx
                forces[j, 1] -= f * dy

        # Boundary repulsion
        margin = 0.05
        forces[:, 0] += np.where(pts[:, 0] < -0.5 + margin, (margin - (pts[:, 0] + 0.5)) * 3.0, 0.0)
        forces[:, 0] -= np.where(pts[:, 0] > 0.5 - margin, (pts[:, 0] - (0.5 - margin)) * 3.0, 0.0)
        forces[:, 1] += np.where(pts[:, 1] < -0.5 + margin, (margin - (pts[:, 1] + 0.5)) * 3.0, 0.0)
        forces[:, 1] -= np.where(pts[:, 1] > 0.5 - margin, (pts[:, 1] - (0.5 - margin)) * 3.0, 0.0)

        pts += step * forces
        pts = np.clip(pts, -0.5, 0.5)

        step *= lr_decay

        r = _get_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    return best_r, best_pts

def _exact_local_search(pts, time_limit, max_iter=5000):
    """
    Exact local search: directly optimize the true minimum radius.
    For each constraint (wall or pair), try to improve it by moving points.
    """
    t_start = time.time()
    pts = pts.copy()
    best_r = _get_radius(pts)
    best_pts = pts.copy()

    for it in range(max_iter):
        if time.time() - t_start > time_limit:
            break

        # Find the current minimum constraint
        r = _get_radius(pts)
        
        # Wall distances
        wall_dists = np.array([
            np.min(pts[:, 0]) + 0.5,
            0.5 - np.max(pts[:, 0]),
            np.min(pts[:, 1]) + 0.5,
            0.5 - np.max(pts[:, 1])
        ])
        min_wall = np.min(wall_dists)
        wall_idx = np.argmin(wall_dists)

        # Pair distances using KDTree
        tree = cKDTree(pts)
        min_pair_dists, min_pair_indices = tree.query(pts, k=2)
        min_pair_dists = min_pair_dists[:, 1]
        min_pair = np.min(min_pair_dists) / 2.0
        pair_idx = np.argmin(min_pair_dists)

        if min_wall <= min_pair:
            # Improve wall constraint
            if wall_idx == 0:  # left wall
                i = np.argmin(pts[:, 0])
                pts[i, 0] += 0.001
            elif wall_idx == 1:  # right wall
                i = np.argmax(pts[:, 0])
                pts[i, 0] -= 0.001
            elif wall_idx == 2:  # bottom wall
                i = np.argmin(pts[:, 1])
                pts[i, 1] += 0.001
            else:  # top wall
                i = np.argmax(pts[:, 1])
                pts[i, 1] -= 0.001
        else:
            # Improve pair constraint
            i = pair_idx
            j = min_pair_indices[pair_idx, 1]
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-10:
                pts[i, 0] += 0.001 * dx / d
                pts[i, 1] += 0.001 * dy / d
                pts[j, 0] -= 0.001 * dx / d
                pts[j, 1] -= 0.001 * dy / d

        pts = np.clip(pts, -0.5, 0.5)

        new_r = _get_radius(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()

    return best_r, best_pts

def _run_optimization(pts, time_limit, betas=None):
    """
    Full optimization pipeline for a single initial configuration:
    1. Force-directed relaxation
    2. L-BFGS-B with increasing beta
    3. Exact local search polish
    """
    if betas is None:
        betas = [10, 20, 50, 100, 200, 500, 1000, 2000]

    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    best_r = _get_radius(pts)
    best_x = x0.copy()

    # Phase 1: Force-directed
    fd_time = min(5.0, time_limit * 0.15)
    if fd_time > 1.0:
        r_fd, pts_fd = _force_directed_phase(pts, time_limit=fd_time)
        if r_fd > best_r:
            best_r = r_fd
            best_x = pts_fd.flatten()
            x0 = pts_fd.flatten()

    # Phase 2: L-BFGS-B with increasing beta
    for beta in betas:
        elapsed = time.time() - t_start
        if elapsed > time_limit:
            break
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break

        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g

        try:
            # Scale maxiter based on remaining time and beta
            if beta >= 1000:
                maxiter = 200
            elif beta >= 200:
                maxiter = 150
            else:
                maxiter = 100
            res = minimize(
                fun_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5, 0.5)

            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()

        except Exception:
            pass

    # Phase 3: Exact local search
    elapsed = time.time() - t_start
    remaining = time_limit - elapsed
    if remaining > 2.0:
        r_els, pts_els = _exact_local_search(
            best_x.reshape((N, 2)),
            time_limit=remaining * 0.8,
            max_iter=5000
        )
        if r_els > best_r:
            best_r = r_els
            best_x = pts_els.flatten()

    return best_r, best_x.reshape((N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Generate multiple initial configurations
    inits = []

    # Init 1: Standard hex with spacing ~0.0732
    pts1 = _generate_hex_initial(0.0732, phase_shift=0.0)
    inits.append(pts1)

    # Init 2: Shifted hex
    pts2 = _generate_hex_initial(0.0732, phase_shift=0.01)
    inits.append(pts2)

    # Init 3: Slightly denser hex (will have overlaps, optimizer fixes)
    pts3 = _generate_hex_initial(0.070, phase_shift=0.0)
    inits.append(pts3)

    # Init 4: Jittered hex
    pts4 = _generate_jittered_hex(0.0732, jitter=0.002, seed=0)
    inits.append(pts4)

    # Init 5: Jittered hex with different seed
    pts5 = _generate_jittered_hex(0.0732, jitter=0.003, seed=1)
    inits.append(pts5)

    # Init 6: Different spacing
    pts6 = _generate_hex_initial(0.071, phase_shift=0.005)
    inits.append(pts6)

    num_inits = len(inits)
    time_per_init = (_TIME_LIMIT - 25) / num_inits  # reserve 25s for final polish

    for i, init_pts in enumerate(inits):
        elapsed = time.time() - t0
        remaining_for_phase = _TIME_LIMIT - 25 - elapsed
        if remaining_for_phase <= 5:
            break

        t_limit = min(time_per_init, remaining_for_phase)
        r_opt, pts_opt = _run_optimization(init_pts, time_limit=t_limit)

        if r_opt > best_r:
            best_r = r_opt
            best_pts = pts_opt

    # Final polish with exact local search
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining_time = _TIME_LIMIT - 3 - elapsed
        if remaining_time > 5:
            # First try high-beta optimization
            betas_polish = [1000, 2000, 5000]
            r_polish, pts_polish = _run_optimization(
                best_pts,
                time_limit=remaining_time * 0.6,
                betas=betas_polish
            )
            if r_polish > best_r:
                best_r = r_polish
                best_pts = pts_polish
            
            # Then exact local search
            elapsed = time.time() - t0
            remaining_time = _TIME_LIMIT - 2 - elapsed
            if remaining_time > 3:
                r_els, pts_els = _exact_local_search(
                    best_pts,
                    time_limit=remaining_time * 0.9,
                    max_iter=10000
                )
                if r_els > best_r:
                    best_r = r_els
                    best_pts = pts_els

    if best_pts is None:
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))

    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
