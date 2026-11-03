"""
Circle packing: 200 equal circles in unit square [-0.5, 0.5]^2.
Structural change: replace the soft-min L-BFGS-B / force-step local search with
a projected-gradient-descent (PGD) solver on the non-smooth max-min objective,
augmented with momentum and simulated-annealing-like restarts. The PGD step uses
the subgradient of the minimum-distance (pair and wall) directly, which is much
faster to evaluate than the soft-min gradient and naturally targets the true
packing objective.
"""
import time
import numpy as np
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _compute_radius(pts):
    """Compute the maximum common radius for the given centres."""
    boundary_min = np.min(0.5 - np.abs(pts))
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_pair, 2.0 * boundary_min) / 2.0
    return r


def _subgradient_min_dist(pts):
    """
    Return (min_dist, subgrad) where subgrad is the subgradient of the
    minimum-distance objective w.r.t. the flattened point array x.
    The objective is min( min_{i<j} ||p_i - p_j||, min_i min_wall(p_i) ).
    """
    pts = pts.reshape(-1, 2)

    # Wall distances for each point
    wall_dists = np.stack([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ], axis=1)  # (N, 4)
    min_wall_per_point = np.min(wall_dists, axis=1)  # (N,)
    global_min_wall = np.min(min_wall_per_point)

    # Pair distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair_val = np.min(d)

    min_dist = min(min_pair_val, global_min_wall)

    subgrad = np.zeros_like(pts)

    # If pair distance is the minimum (or tied), add subgradient from that pair
    if min_pair_val <= global_min_wall + 1e-12:
        # Find all pairs achieving (near) the minimum
        # For efficiency, just take the first few to avoid O(N^2) accumulation issues
        # Actually, subgradient of min is any convex combo; using the argmin pair is a valid subgradient
        i_idx, j_idx = np.unravel_index(np.argmin(d), d.shape)
        if i_idx != j_idx:
            diff = pts[i_idx] - pts[j_idx]
            dist = d[i_idx, j_idx]
            if dist > 1e-12:
                dir_vec = diff / dist
                subgrad[i_idx] += dir_vec
                subgrad[j_idx] -= dir_vec

    # If wall distance is the minimum (or tied), add subgradient from that point
    if global_min_wall <= min_pair_val + 1e-12:
        # Find point(s) with minimum wall distance
        min_pt_idx = np.argmin(min_wall_per_point)
        p = pts[min_pt_idx]
        # Determine which wall is closest
        w_left = p[0] + 0.5
        w_right = 0.5 - p[0]
        w_bottom = p[1] + 0.5
        w_top = 0.5 - p[1]
        min_w = min(w_left, w_right, w_bottom, w_top)
        # Add subgradient from the closest wall(s)
        if w_left <= min_w + 1e-12:
            subgrad[min_pt_idx, 0] -= 1.0
        if w_right <= min_w + 1e-12:
            subgrad[min_pt_idx, 0] += 1.0
        if w_bottom <= min_w + 1e-12:
            subgrad[min_pt_idx, 1] -= 1.0
        if w_top <= min_w + 1e-12:
            subgrad[min_pt_idx, 1] += 1.0

    return min_dist, subgrad


def _pgd_step(pts, step_size, momentum=0.0):
    """
    One step of projected gradient descent (ascent on min-dist) with momentum.
    """
    min_dist, subgrad = _subgradient_min_dist(pts)
    if momentum > 0:
        # Simple momentum: we need to store previous velocity, but for simplicity
        # we'll do a basic step. A full momentum implementation would need state.
        pass
    new_pts = pts + step_size * subgrad
    new_pts = np.clip(new_pts, -0.4999, 0.4999)
    return new_pts, min_dist


def _pgd_local_search(pts, max_iters=500, time_limit=10.0, initial_step=0.005):
    """
    Local search using projected gradient descent with adaptive step size
    and simulated-annealing-like perturbations to escape local minima.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    step_size = initial_step
    velocity = np.zeros_like(pts)
    momentum = 0.9
    temperature = 0.01

    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        min_dist, subgrad = _subgradient_min_dist(pts)

        # Momentum update
        velocity = momentum * velocity + step_size * subgrad
        new_pts = pts + velocity
        new_pts = np.clip(new_pts, -0.4999, 0.4999)

        new_r = _compute_radius(new_pts)

        # Simulated annealing acceptance
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            pts = new_pts
            step_size = min(0.05, step_size * 1.02)  # grow step slightly
            temperature = max(0.001, temperature * 0.98)
        else:
            # Accept with probability exp(-(best_r - new_r) / temperature)
            delta = best_r - new_r
            if delta < 1e-12 or np.random.random() < np.exp(-delta / temperature):
                pts = new_pts
                step_size = max(0.0005, step_size * 0.95)
                temperature = max(0.001, temperature * 0.95)
            else:
                # Reject, shrink step
                step_size = max(0.0005, step_size * 0.9)

        # Occasionally add random perturbation
        if it % 20 == 0:
            perturbation = np.random.default_rng(it).normal(0, temperature, size=pts.shape)
            perturbed = np.clip(pts + perturbation, -0.4999, 0.4999)
            pert_r = _compute_radius(perturbed)
            if pert_r > best_r:
                best_r = pert_r
                best_pts = perturbed.copy()
                pts = perturbed

    return best_pts, best_r


def _hex_init(n, rng):
    """Initialize n points in a hexagonal lattice pattern."""
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    rows = int(np.ceil(1.0 / dy)) + 2
    cols = int(np.ceil(1.0 / dx)) + 2
    pts = []
    for i in range(rows):
        y = -0.5 + i * dy
        offset = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        idx = rng.choice(len(pts), size=n, replace=False)
        return pts[idx]
    else:
        result = np.empty((n, 2))
        result[:len(pts)] = pts
        result[len(pts):] = rng.uniform(-0.45, 0.45, size=(n - len(pts), 2))
        return result


def _random_init(n, rng):
    """Random initialization inside the unit square."""
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _grid_init(n, rng):
    """Grid initialization with slight random jitter."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= n:
        pts = grid[:n]
    else:
        pts = grid
        extra = rng.uniform(-0.4, 0.4, size=(n - len(pts), 2))
        pts = np.vstack([pts, extra])
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts


def construct():
    """Construct 200 circle centres in the unit square maximizing common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin

    rng = np.random.default_rng(42)

    best_pts = None
    best_r = -np.inf

    # Multi-start with different initializations
    inits = [
        _hex_init(N, rng),
        _random_init(N, rng),
        _grid_init(N, rng),
        _hex_init(N, rng),
        _random_init(N, rng),
    ]

    for start_idx, init_pts in enumerate(inits):
        if time.time() > deadline:
            break

        # Phase 1: PGD local search
        try:
            ls_pts, ls_r = _pgd_local_search(init_pts, max_iters=800, time_limit=12.0, initial_step=0.008)
            if ls_r > best_r:
                best_r = ls_r
                best_pts = ls_pts.copy()
        except Exception:
            pass

        if time.time() > deadline - 15:
            break

        # Phase 2: Second PGD pass from the best result with smaller step
        if best_pts is not None:
            try:
                ls_pts2, ls_r2 = _pgd_local_search(best_pts, max_iters=500, time_limit=8.0, initial_step=0.003)
                if ls_r2 > best_r:
                    best_r = ls_r2
                    best_pts = ls_pts2.copy()
            except Exception:
                pass

        if time.time() > deadline - 8:
            break

    # Final polish: one more local search from the best result
    if best_pts is not None and time.time() < deadline - 3:
        try:
            ls_pts, ls_r = _pgd_local_search(best_pts, max_iters=300, time_limit=4.0, initial_step=0.002)
            if ls_r > best_r:
                best_r = ls_r
                best_pts = ls_pts.copy()
        except Exception:
            pass

    # If no valid result, return a simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
