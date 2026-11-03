"""
Different strategy: Simulated Annealing with spatial hashing for O(n) neighbor
lookup, combined with a final gradient-based local refinement.

Key structural differences from the current MD approach:
1. Uses spatial hashing (grid) instead of all-pairs O(n^2) distance checks
2. Simulated annealing with geometric cooling instead of velocity Verlet
3. Multi-restart with diverse initializations
4. Final local optimization using gradient ascent on the minimum constraint
"""
import numpy as np
import time
import math

N = 300
TIME_LIMIT = 115.0  # safety margin before 150s kill


def _hex_init(n=N, spacing=0.106, jitter=0.0):
    """Dense hexagonal lattice clipped to unit disk, keep n closest to center."""
    k = int(1.2 / spacing) + 3
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * math.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        return _hex_init(n, spacing * 0.92)
    order = np.argsort(r, kind="stable")
    pts = pts[order[:n]].copy()
    if jitter > 0:
        rng = np.random.default_rng(42)
        pts += rng.normal(0, jitter, size=pts.shape)
        # Clamp inside
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]
    return pts


def _build_grid(pts, cell_size):
    """Build spatial hash grid. Returns dict mapping (ix, iy) -> list of indices."""
    grid = {}
    n = len(pts)
    for i in range(n):
        ix = int(math.floor(pts[i, 0] / cell_size))
        iy = int(math.floor(pts[i, 1] / cell_size))
        key = (ix, iy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid


def _get_neighbors(i, pts, grid, cell_size):
    """Get indices of points in neighboring cells of point i."""
    ix = int(math.floor(pts[i, 0] / cell_size))
    iy = int(math.floor(pts[i, 1] / cell_size))
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            key = (ix + dx, iy + dy)
            if key in grid:
                neighbors.extend(grid[key])
    # Remove self
    if i in neighbors:
        neighbors.remove(i)
    return neighbors


def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)


def _anneal(pts, start_time, time_limit, T0=0.008, T_end=1e-6, cooling=0.9995, seed=42):
    """
    Simulated annealing with spatial hashing.
    Moves are random displacements of single points.
    Accept if it improves the minimum constraint, or with probability exp(-delta/T).
    """
    rng = np.random.default_rng(seed)
    pos = pts.copy()
    n = len(pos)
    
    # Estimate cell size from current packing
    cell_size = 0.12
    
    best_pos = pos.copy()
    best_score = _score(pos)
    
    T = T0
    no_improve = 0
    max_no_improve = 5000
    it = 0
    
    # Track the minimum distance and boundary for fast evaluation
    # We'll recompute score periodically, but use a fast check for moves
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break
        if T < T_end:
            break
        
        # Pick a random point to move
        i = rng.integers(0, n)
        old_pos = pos[i].copy()
        
        # Random displacement
        angle = rng.uniform(0, 2 * math.pi)
        mag = rng.uniform(0, 0.02)
        dx = mag * math.cos(angle)
        dy = mag * math.sin(angle)
        new_pos = old_pos + np.array([dx, dy])
        
        # Check if new position is inside unit circle
        new_rad = math.hypot(new_pos[0], new_pos[1])
        if new_rad > 1.0:
            # Reflect or skip
            pos[i] = new_pos / new_rad
            new_rad = 1.0
            new_pos = pos[i]
        
        # Compute change in minimum constraint
        # Only need to check neighbors of point i
        grid = _build_grid(pos, cell_size)
        neighbors = _get_neighbors(i, pos, grid, cell_size)
        
        min_dist_before = 1.0
        # Find min distance involving i before move
        for j in neighbors:
            d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < min_dist_before:
                min_dist_before = d
        # Also check boundary for i
        bound_before = 1.0 - math.hypot(old_pos[0], old_pos[1])
        
        # Apply move
        pos[i] = new_pos
        new_old_rad = math.hypot(new_pos[0], new_pos[1])
        bound_after = 1.0 - new_old_rad
        
        # Recompute min distance involving i after move
        min_dist_after = 1.0
        grid = _build_grid(pos, cell_size)
        neighbors = _get_neighbors(i, pos, grid, cell_size)
        for j in neighbors:
            d = math.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < min_dist_after:
                min_dist_after = d
        
        # The score is min(min_pair/2, min_bound)
        # We only changed constraints involving i, so we need global min
        # For speed, we approximate: if the local min involving i is still above
        # the previous global min, the score hasn't changed.
        
        # Periodic full score evaluation
        if it % 100 == 0:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                no_improve = 0
            else:
                no_improve += 100
        
        # Accept/reject based on local change
        # Compute local score change
        local_before = min(min_dist_before / 2.0, bound_before)
        local_after = min(min_dist_after / 2.0, bound_after)
        
        delta = local_after - local_before
        if delta >= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            it += 1
        else:
            # Reject
            pos[i] = old_pos
            it += 1
        
        # Cool
        T *= cooling
        
        # Early termination if settled
        if it > 20000 and T < 1e-5:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
            break
    
    final_s = _score(pos)
    if final_s > best_score:
        best_score = final_s
        best_pos = pos.copy()
    
    return best_pos, best_score


def _local_refine(pts, start_time, time_limit):
    """
    Local refinement: iteratively move the most constrained point to improve
    the minimum constraint.
    """
    n = len(pts)
    pos = pts.copy()
    best_pos = pos.copy()
    best_score = _score(pos)
    it = 0
    no_improve = 0
    max_no_improve = 200
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        if no_improve > max_no_improve:
            break
        
        # Compute all pairwise distances and boundary distances
        d2 = np.sum(pos * pos, axis=1)[:, None] - 2.0 * (pos @ pos.T) + np.sum(pos * pos, axis=1)[None, :]
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(np.maximum(d2, 0.0))
        
        rad = np.hypot(pos[:, 0], pos[:, 1])
        r_bound = 1.0 - rad
        
        # Find the minimum constraint
        min_pair_val = np.min(d) / 2.0
        min_bound_val = np.min(r_bound)
        current_score = min(min_pair_val, min_bound_val)
        
        # Find which constraint is limiting
        if min_pair_val <= min_bound_val:
            # Pair constraint is limiting
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx[0], idx[1]
            if i > j:
                i, j = j, i
            # Move point i away from j
            diff = pos[i] - pos[j]
            dist = math.hypot(diff[0], diff[1])
            if dist > 1e-10:
                unit = diff / dist
                # Try to move i in the direction away from j
                step = 0.005
                new_i = pos[i] + unit * step
                # Check if still inside
                new_rad = math.hypot(new_i[0], new_i[1])
                if new_rad <= 1.0:
                    pos[i] = new_i
                else:
                    # Try smaller step
                    step = 0.002
                    new_i = pos[i] - unit * step  # Try opposite? No, we want away
                    new_i = pos[i] + unit * step
                    new_rad = math.hypot(new_i[0], new_i[1])
                    if new_rad <= 1.0:
                        pos[i] = new_i
            it += 1
        else:
            # Boundary constraint is limiting
            i = np.argmin(r_bound)
            # Move point i toward center
            rad_i = rad[i]
            if rad_i > 1e-10:
                inward = -pos[i] / rad_i
                step = 0.005
                new_i = pos[i] + inward * step
                pos[i] = new_i
            it += 1
        
        # Check improvement
        if it % 10 == 0:
            s = _score(pos)
            if s > best_score:
                best_score = s
                best_pos = pos.copy()
                no_improve = 0
            else:
                no_improve += 10
    
    return best_pos, best_score


def construct():
    start_time = time.time()
    
    # Multiple diverse initializations
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Try several initializations with annealing
    initializations = [
        _hex_init(N, spacing=0.106, jitter=0.0),
        _hex_init(N, spacing=0.108, jitter=0.005),
        _hex_init(N, spacing=0.104, jitter=0.01),
    ]
    
    time_per_anneal = TIME_LIMIT * 0.4 / len(initializations)
    
    for k, init_pts in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < time_per_anneal * 0.5:
            break
        
        # Clamp inside
        rad = np.hypot(init_pts[:, 0], init_pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            init_pts[out] = init_pts[out] / rad[out, None]
        
        try:
            anneal_pts, anneal_score = _anneal(
                init_pts, start_time, TIME_LIMIT - (TIME_LIMIT - remaining),
                T0=0.008, T_end=1e-6, cooling=0.9995, seed=42 + k
            )
            if anneal_score > best_score:
                best_score = anneal_score
                best_pts = anneal_pts.copy()
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0]
        best_score = _score(best_pts)
    
    # Phase 2: Local refinement
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 5.0:
        try:
            refine_pts, refine_score = _local_refine(best_pts, start_time, TIME_LIMIT)
            if refine_score > best_score:
                best_score = refine_score
                best_pts = refine_pts
        except Exception:
            pass
    
    # Final safety
    best_pts = np.asarray(best_pts, dtype=float)
    if best_pts.shape != (N, 2):
        best_pts = _hex_init(N)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
    
    return best_pts
