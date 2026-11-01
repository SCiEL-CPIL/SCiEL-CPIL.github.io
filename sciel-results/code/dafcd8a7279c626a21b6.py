"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N_CIRCLES = 200
HALF_SIDE = 0.5


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0

    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.08)
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)

    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def _min_clearance(centers, i, tree, max_query_r=0.15):
    """Minimum clearance of circle i to any other circle center or wall."""
    ci = centers[i]
    # wall clearance
    wall = min(HALF_SIDE - ci[0], ci[0] + HALF_SIDE,
               HALF_SIDE - ci[1], ci[1] + HALF_SIDE)
    # neighbor clearance
    neighbors, _ = tree.query(ci, k=8, distance_upper_bound=max_query_r)
    if isinstance(neighbors, np.ndarray) and neighbors.ndim == 1:
        neighbors = neighbors.reshape(-1, 1)
    if neighbors.size == 0:
        return wall
    dists = np.linalg.norm(centers[neighbors] - ci, axis=1)
    return min(wall, dists.min())


def _greedy_insertion(n, seed=0):
    """Sequentially place n circles, each at a candidate that maximizes min clearance."""
    rng = np.random.RandomState(seed)
    centers = np.empty((n, 2))
    placed = 0
    # coarse candidate grid
    m = 40
    ticks = (np.arange(m) + 0.5) / m - 0.5
    gx, gy = np.meshgrid(ticks, ticks)
    candidates = np.column_stack([gx.ravel(), gy.ravel()])
    # also add a finer grid near center for later placements
    m2 = 60
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    gx2, gy2 = np.meshgrid(ticks2, ticks2)
    candidates2 = np.column_stack([gx2.ravel(), gy2.ravel()])

    for k in range(n):
        if placed == 0:
            centers[0] = [0.0, 0.0]
            placed = 1
            continue

        # Choose candidate set: for early placements use coarse, later use fine + jitter
        if placed < n // 4:
            cand = candidates
        else:
            cand = candidates2

        # Add a few random jittered candidates for diversity
        jitter = rng.uniform(-0.01, 0.01, size=(50, 2))
        jitter = np.clip(jitter, -HALF_SIDE + 1e-9, HALF_SIDE - 1e-9)
        cand = np.vstack([cand, jitter])

        # Compute min clearance for each candidate to already-placed circles and walls
        placed_pts = centers[:placed]
        # wall clearance for candidates
        wall_c = np.minimum(np.minimum(HALF_SIDE - cand[:, 0], cand[:, 0] + HALF_SIDE),
                            np.minimum(HALF_SIDE - cand[:, 1], cand[:, 1] + HALF_SIDE))
        # neighbor clearance: for each candidate, min dist to placed_pts
        # Use cKDTree on placed points
        tree_p = cKDTree(placed_pts)
        d, _ = tree_p.query(cand, k=1)
        clearance = np.minimum(wall_c, d)
        best_idx = np.argmax(clearance)
        centers[placed] = cand[best_idx]
        placed += 1

    return centers


def _repair_circle(centers, i, tree, max_iter=30):
    """Nelder-Mead optimize circle i to maximize its min clearance to neighbours/walls."""
    ci = centers[i].copy()

    def neg_clearance(x):
        x = np.clip(x, -HALF_SIDE + 1e-9, HALF_SIDE - 1e-9)
        wall = min(HALF_SIDE - x[0], x[0] + HALF_SIDE,
                   HALF_SIDE - x[1], x[1] + HALF_SIDE)
        neighbors, _ = tree.query(x, k=8, distance_upper_bound=0.15)
        if isinstance(neighbors, np.ndarray) and neighbors.ndim == 1:
            neighbors = neighbors.reshape(-1, 1)
        if neighbors.size == 0:
            return -wall
        dists = np.linalg.norm(centers[neighbors] - x, axis=1)
        return -min(wall, dists.min())

    # Simple Nelder-Mead with small initial simplex
    step = 0.003
    x0 = ci.copy()
    simplex = [x0.copy(),
               x0 + np.array([step, 0.0]),
               x0 + np.array([0.0, step])]
    fvals = [neg_clearance(v) for v in simplex]

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[o] for o in order]
        fvals = [fvals[o] for o in order]
        # If improvement is tiny, stop
        if abs(fvals[0] - fvals[-1]) < 1e-12:
            break
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = neg_clearance(xr)
        if fr < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = neg_clearance(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = neg_clearance(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for j in range(1, len(simplex)):
                    simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                    fvals[j] = neg_clearance(simplex[j])
    # Take best point
    best = min(simplex, key=lambda v: neg_clearance(v))
    centers[i] = np.clip(best, -HALF_SIDE + 1e-9, HALF_SIDE - 1e-9)
    return centers


def _local_repair(centers, max_passes=30):
    """Iteratively repair circles in worst-first order until no improvement."""
    current = centers.copy()
    best_r = compute_radius(current)
    for _ in range(max_passes):
        tree = cKDTree(current)
        # Compute clearance for each circle
        clearances = np.array([_min_clearance(current, i, tree) for i in range(N_CIRCLES)])
        # Sort by worst clearance
        order = np.argsort(clearances)
        improved = False
        for i in order:
            before = clearances[i]
            _repair_circle(current, i, tree, max_iter=25)
            # Recompute this circle's clearance
            after = _min_clearance(current, i, tree)
            if after > before + 1e-12:
                improved = True
        # Check global radius
        new_r = compute_radius(current)
        if new_r > best_r:
            best_r = new_r
        if not improved:
            break
    return current, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    np.random.seed(42)

    # Try multiple seeds for greedy insertion, keep best
    best_config = None
    best_r = 0.0

    for seed in [0, 1, 2]:
        if time.time() > deadline - 30:
            break
        cfg = _greedy_insertion(N_CIRCLES, seed=seed)
        r = compute_radius(cfg)
        if r > best_r:
            best_r = r
            best_config = cfg

    # Local repair with worst-first Nelder-Mead
    if time.time() < deadline - 15:
        repaired, rep_r = _local_repair(best_config, max_passes=20)
        if rep_r > best_r:
            best_r = rep_r
            best_config = repaired

    # A second, lighter repair pass if time remains
    if time.time() < deadline - 5:
        repaired2, rep2_r = _local_repair(best_config, max_passes=10)
        if rep2_r > best_r:
            best_r = rep2_r
            best_config = repaired2

    # Final validity check and fallback
    final_r = compute_radius(best_config)
    if final_r <= 0 or best_config is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N_CIRCLES)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_config = grid[:N_CIRCLES]

    return best_config
# EVOLVE-BLOCK-END
