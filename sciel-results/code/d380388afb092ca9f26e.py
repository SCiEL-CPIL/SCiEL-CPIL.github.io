"""
Packing 50 equal circles in a unit square.
Strategy: sequential hard-constrained placement + force-directed relaxation + local polish.
"""
import numpy as np
from scipy.spatial import cKDTree
import time
import random

N = 50
HALF = 0.5

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if len(centres) != N:
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    if np.any(np.abs(centres) > HALF + 1e-9):
        return 0.0
    # Min pairwise distance
    tree = cKDTree(centres)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = 1.0  # very large
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    # Min distance to walls
    dist_to_walls = np.minimum(HALF - np.abs(centres[:, 0]), HALF - np.abs(centres[:, 1]))
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)


def _min_radius_fast(centres):
    """Fast radius computation using KD-tree for close pairs only."""
    tree = cKDTree(centres)
    pairs = tree.query_pairs(r=0.3)  # only need close pairs
    if len(pairs) == 0:
        min_pair_dist = 1.0
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair_dist:
                min_pair_dist = d
    dist_to_walls = np.minimum(HALF - np.abs(centres[:, 0]), HALF - np.abs(centres[:, 1]))
    min_wall_dist = np.min(dist_to_walls)
    return min(min_pair_dist / 2.0, min_wall_dist)


def _sequential_placement(r, n_trials=200, seed=42):
    """
    Place N circles of radius r sequentially using a spatial hash.
    Returns centres array or None if placement fails.
    """
    rng = np.random.RandomState(seed)
    placed = []
    # Use a grid-based spatial hash for O(1) neighbor lookup
    cell_size = 2 * r * 1.5
    nx = int(np.ceil(1.0 / cell_size)) + 2
    ny = int(np.ceil(1.0 / cell_size)) + 2
    grid = {}
    
    def get_cell(x, y):
        cx = int((x + HALF) / cell_size)
        cy = int((y + HALF) / cell_size)
        return (cx, cy)
    
    for i in range(N):
        success = False
        for trial in range(n_trials):
            x = rng.uniform(-HALF + r, HALF - r)
            y = rng.uniform(-HALF + r, HALF - r)
            # Check against all placed circles (brute force for N=50 is fine)
            ok = True
            for j in range(len(placed)):
                dx = x - placed[j][0]
                dy = y - placed[j][1]
                if dx * dx + dy * dy < (2 * r) * (2 * r) - 1e-12:
                    ok = False
                    break
            if ok:
                placed.append((x, y))
                cx, cy = get_cell(x, y)
                grid.setdefault((cx, cy), []).append(len(placed) - 1)
                success = True
                break
        if not success:
            return None
    
    return np.array(placed)


def _force_directed_relax(centres, r, max_iters=5000, dt=0.002, seed=42):
    """
    Force-directed relaxation: circles repel each other (if too close) and 
    are pushed inward if too close to walls. Only accept moves that maintain
    all distances >= 2r and all within bounds.
    """
    rng = np.random.RandomState(seed)
    c = centres.copy()
    n = len(c)
    
    for it in range(max_iters):
        # Compute forces
        forces = np.zeros_like(c)
        # Pairwise repulsion for pairs closer than 2r (with soft core)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                d2 = dx * dx + dy * dy
                d = np.sqrt(d2)
                if d < 2 * r and d > 1e-10:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - d
                    f = 10.0 * overlap / d
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
                elif d < 2 * r * 1.1 and d > 1e-10:
                    # Mild repulsion for near-contact pairs to spread out
                    f = 0.1 / d
                    forces[i, 0] += f * dx
                    forces[i, 1] += f * dy
                    forces[j, 0] -= f * dx
                    forces[j, 1] -= f * dy
        
        # Wall repulsion
        for i in range(n):
            # Left wall
            dist_l = c[i, 0] + HALF
            if dist_l < r:
                forces[i, 0] += 10.0 * (r - dist_l)
            # Right wall
            dist_r = HALF - c[i, 0]
            if dist_r < r:
                forces[i, 0] -= 10.0 * (r - dist_r)
            # Bottom wall
            dist_b = c[i, 1] + HALF
            if dist_b < r:
                forces[i, 1] += 10.0 * (r - dist_b)
            # Top wall
            dist_t = HALF - c[i, 1]
            if dist_t < r:
                forces[i, 1] -= 10.0 * (r - dist_t)
        
        # Proposed move
        new_c = c + dt * forces
        new_c = np.clip(new_c, -HALF + r, HALF - r)
        
        # Check validity: all pairs >= 2r
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = new_c[i, 0] - new_c[j, 0]
                dy = new_c[i, 1] - new_c[j, 1]
                if dx * dx + dy * dy < (2 * r) * (2 * r) - 1e-10:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            c = new_c
        else:
            # Try smaller step
            new_c = c + 0.1 * dt * forces
            new_c = np.clip(new_c, -HALF + r, HALF - r)
            valid = True
            for i in range(n):
                for j in range(i + 1, n):
                    dx = new_c[i, 0] - new_c[j, 0]
                    dy = new_c[i, 1] - new_c[j, 1]
                    if dx * dx + dy * dy < (2 * r) * (2 * r) - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                c = new_c
    
    return c


def _local_polish(centres, r, max_iters=2000, seed=42):
    """
    Local polish: try small random moves on individual circles, accept if
    all constraints (pairwise >= 2r, within bounds) are maintained.
    This can improve the minimum distance beyond 2r.
    """
    rng = np.random.RandomState(seed)
    c = centres.copy()
    n = len(c)
    
    # Current minimum margin
    def min_margin(arr):
        tree = cKDTree(arr)
        pairs = tree.query_pairs(r=2 * r * 1.5)
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(arr[i] - arr[j])
            if d < min_d:
                min_d = d
        wall_d = np.min(np.minimum(HALF - np.abs(arr[:, 0]), HALF - np.abs(arr[:, 1])))
        return min(min_d, 2 * wall_d)
    
    current_margin = min_margin(c)
    
    for it in range(max_iters):
        i = rng.randint(n)
        dx = rng.normal(0, 0.005)
        dy = rng.normal(0, 0.005)
        old_x, old_y = c[i, 0], c[i, 1]
        c[i, 0] = np.clip(old_x + dx, -HALF + r, HALF - r)
        c[i, 1] = np.clip(old_y + dy, -HALF + r, HALF - r)
        
        # Check validity
        valid = True
        for j in range(n):
            if j == i:
                continue
            ddx = c[i, 0] - c[j, 0]
            ddy = c[i, 1] - c[j, 1]
            if ddx * ddx + ddy * ddy < (2 * r) * (2 * r) - 1e-12:
                valid = False
                break
        
        if valid:
            new_margin = min_margin(c)
            if new_margin > current_margin:
                current_margin = new_margin
            else:
                # Keep the move if it doesn't hurt (for diversity)
                pass
        else:
            c[i, 0] = old_x
            c[i, 1] = old_y
    
    return c


def _try_radius(r, time_budget, seed=42):
    """
    Try to find a valid packing of N circles with radius r.
    Returns (centres, success).
    """
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Multiple random restarts for sequential placement
    for attempt in range(20):
        if time.time() - start > time_budget:
            break
        centres = _sequential_placement(r, n_trials=500, seed=seed + attempt * 1000)
        if centres is not None:
            # Relax to improve minimum distance
            remaining = time_budget - (time.time() - start)
            if remaining > 2:
                centres = _force_directed_relax(centres, r, max_iters=2000, dt=0.002, seed=seed + attempt * 2000)
            if time.time() - start < time_budget - 1:
                centres = _local_polish(centres, r, max_iters=500, seed=seed + attempt * 3000)
            # Verify
            actual_r = _min_radius_fast(centres)
            if actual_r >= r - 1e-9:
                return centres, True
            elif actual_r > r * 0.95:
                # Close enough, use it
                return centres, True
    
    return None, False


def _bisection_search(r_low, r_high, time_budget, seed=42):
    """
    Binary search for the maximum radius.
    """
    start = time.time()
    best_centres = None
    best_r = r_low
    
    # First, verify r_low works
    c, ok = _try_radius(r_low, time_budget * 0.3, seed)
    if ok:
        best_centres = c
        best_r = r_low
    
    # Binary search
    n_iters = 12
    for i in range(n_iters):
        if time.time() - start > time_budget * 0.8:
            break
        r_mid = (r_low + r_high) / 2
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            break
        
        c, ok = _try_radius(r_mid, remaining * 0.4, seed + i * 5000)
        if ok:
            best_centres = c
            best_r = r_mid
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_centres, best_r


def _grid_init(n_cols=None):
    """Generate a grid initialization as fallback."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    random.seed(42)
    
    # Start with a grid as baseline
    best_centres = _grid_init()
    best_r = compute_radius(best_centres)
    
    # Phase 1: Sequential placement + relaxation at various radii
    # Try radii from low to high
    r_candidates = [0.06, 0.065, 0.068, 0.070, 0.071, 0.0713, 0.0714]
    
    for r_try in r_candidates:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        c, ok = _try_radius(r_try, remaining * 0.6, seed=42 + int(r_try * 10000))
        if ok and c is not None:
            actual_r = _min_radius_fast(c)
            if actual_r > best_r:
                best_r = actual_r
                best_centres = c.copy()
    
    # Phase 2: Bisection search around the best found radius
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        r_low = best_r * 0.95
        r_high = min(best_r * 1.02, 0.072)
        c, r_found = _bisection_search(r_low, r_high, remaining * 0.7, seed=99999)
        if c is not None:
            actual_r = _min_radius_fast(c)
            if actual_r > best_r:
                best_r = actual_r
                best_centres = c.copy()
    
    # Phase 3: Final local polish on the best result
    if time.time() < deadline - 5:
        r_current = _min_radius_fast(best_centres)
        # Polish with a slightly smaller constraint to allow improvement
        polished = _local_polish(best_centres, r_current * 0.99, max_iters=2000, seed=77777)
        polished_r = _min_radius_fast(polished)
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished
    
    # Final validation
    best_centres = np.clip(best_centres, -HALF, HALF)
    if np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
        best_r = compute_radius(best_centres)
    
    return best_centres
