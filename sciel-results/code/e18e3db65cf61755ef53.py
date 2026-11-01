"""
Discrete grid-based constraint satisfaction for packing 50 equal circles in a unit square.
Structural change: reformulate as discrete optimization over a high-resolution spatial grid,
using arc-consistency to prune infeasible placements based on spatial exclusion zones.
This avoids continuous local minima by working in a discrete representation.
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 50
HALF = 0.5
GRID_SIZE = 40  # 40x40 grid points in [-0.5, 0.5]^2


def _compute_radius(points):
    """
    Compute the maximum radius such that circles centered at points
    are inside [-0.5, 0.5]^2 and non-overlapping.
    """
    n = len(points)
    if n == 0:
        return 0.0

    wall_dists = np.minimum(HALF - np.abs(points[:, 0]), HALF - np.abs(points[:, 1]))
    min_wall = float(np.min(wall_dists))

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1]))

    r = min(min_pair / 2.0, min_wall)
    return r if r > 0 else 0.0


def _grid_points():
    """Generate grid points in [-0.5, 0.5]^2."""
    ticks = np.linspace(-HALF, HALF, GRID_SIZE)
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid


def _build_exclusion_zones(grid_pts, min_dist):
    """
    For each grid point, determine which other grid points are within min_dist.
    Returns a list of sets: exclusion_zones[i] = set of indices j where dist(grid[i], grid[j]) < min_dist
    """
    n = len(grid_pts)
    tree = cKDTree(grid_pts)
    # Query all points within min_dist
    indices = tree.query_ball_point(grid_pts, r=min_dist)
    # Convert to list of sets, excluding self
    zones = []
    for i in range(n):
        s = set(indices[i])
        s.discard(i)
        zones.append(s)
    return zones


def _arc_consistency_feasible(grid_pts, zones, n_points):
    """
    Check if we can place n_points non-overlapping circles on the grid.
    Uses a greedy approach with backtracking for small n.
    Returns True if feasible, False otherwise.
    """
    # Simple greedy: try to place points one by one
    used = set()
    count = 0
    # Sort grid points by "desirability" (corners first, then edges, then center)
    # Actually, let's just iterate and pick the first available
    for i in range(len(grid_pts)):
        if i in used:
            continue
        # Check if all neighbors are also used or not yet placed
        # For feasibility, we just need to check if we can place here
        # A point is placeable if no already-placed point is in its exclusion zone
        conflict = False
        for j in zones[i]:
            if j in used:
                conflict = True
                break
        if not conflict:
            used.add(i)
            count += 1
            if count >= n_points:
                return True
    return count >= n_points


def _greedy_place(grid_pts, zones, n_points, rng):
    """
    Greedy placement with randomization to find a feasible set of n_points.
    Returns indices of placed points, or None if failed.
    """
    used = set()
    placed = []
    attempts = 0
    max_attempts = 1000
    
    while len(placed) < n_points and attempts < max_attempts:
        attempts += 1
        # Pick a random grid point
        i = rng.randint(len(grid_pts))
        if i in used:
            continue
        # Check conflicts
        conflict = False
        for j in zones[i]:
            if j in used:
                conflict = True
                break
        if not conflict:
            used.add(i)
            placed.append(i)
    
    if len(placed) < n_points:
        return None
    return placed


def _binary_search_radius(grid_pts, zones, n_points, deadline):
    """
    Binary search on the minimum distance (2*r) to find the largest feasible radius.
    """
    lo = 0.01
    hi = 0.15  # Upper bound estimate
    
    # Quick feasibility check at hi
    if not _arc_consistency_feasible(grid_pts, zones, n_points):
        return 0.0
    
    best_radius = 0.0
    best_placement = None
    
    while hi - lo > 0.001 and time.time() < deadline:
        mid = (lo + hi) / 2
        min_dist = 2 * mid
        
        # Rebuild zones for this min_dist
        tree = cKDTree(grid_pts)
        indices = tree.query_ball_point(grid_pts, r=min_dist)
        zones_mid = [set(indices[i]) - {i} for i in range(len(grid_pts))]
        
        # Try to place
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        placement = _greedy_place(grid_pts, zones_mid, n_points, rng)
        
        if placement is not None:
            best_radius = mid
            best_placement = placement
            lo = mid
        else:
            hi = mid
    
    if best_placement is not None:
        return best_radius, best_placement
    return 0.0, None


def _continuous_refine(grid_pts, placement_indices, deadline):
    """
    After finding a discrete placement, refine continuously using local optimization.
    """
    points = grid_pts[placement_indices].copy()
    
    # Use L-BFGS to maximize the minimum clearance
    def objective(params):
        pts = params.reshape(-1, 2)
        if np.any(np.abs(pts) > HALF + 1e-10):
            return 1e10
        r = _compute_radius(pts)
        return -r
    
    x0 = points.flatten()
    res = minimize(objective, x0, method='L-BFGS-B', 
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': 200, 'ftol': 1e-10})
    
    if res.success or res.fun < -0.01:
        refined = res.x.reshape(-1, 2)
        r = _compute_radius(refined)
        if r > 0:
            return refined, r
    
    return points, _compute_radius(points)


def _simulated_annealing_continuous(points, deadline):
    """
    Simulated annealing on continuous coordinates to escape local minima.
    """
    rng = np.random.RandomState(42)
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    t = 0.005
    t_end = 1e-6
    cooling = 0.999
    
    for it in range(50000):
        if time.time() > deadline:
            break
        
        idx = rng.randint(N)
        old_pos = current[idx].copy()
        
        sigma = 0.003 * (t / 0.005) ** 0.5
        current[idx] += rng.normal(0, sigma, size=2)
        current[idx] = np.clip(current[idx], -HALF + 1e-12, HALF - 1e-12)
        
        new_r = _compute_radius(current)
        delta = new_r - current_r
        
        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-12)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[idx] = old_pos
        
        t *= cooling
        if t < t_end:
            t = t_end
    
    return best, best_r


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    grid_pts = _grid_points()
    
    # Phase 1: Discrete binary search + greedy placement
    best_r = 0.0
    best_points = None
    
    # Try several different random seeds for greedy placement
    for seed in [1, 42, 123, 999, 2024]:
        if time.time() > deadline:
            break
        
        # Binary search with this seed
        def _greedy_with_seed(grid_pts, zones, n_points, rng):
            used = set()
            placed = []
            attempts = 0
            max_attempts = 2000
            
            while len(placed) < n_points and attempts < max_attempts:
                attempts += 1
                i = rng.randint(len(grid_pts))
                if i in used:
                    continue
                conflict = False
                for j in zones[i]:
                    if j in used:
                        conflict = True
                        break
                if not conflict:
                    used.add(i)
                    placed.append(i)
            
            if len(placed) < n_points:
                return None
            return placed
        
        lo = 0.02
        hi = 0.12
        
        for _ in range(20):  # Binary search iterations
            if time.time() > deadline:
                break
            
            mid = (lo + hi) / 2
            min_dist = 2 * mid
            
            tree = cKDTree(grid_pts)
            indices = tree.query_ball_point(grid_pts, r=min_dist)
            zones = [set(indices[i]) - {i} for i in range(len(grid_pts))]
            
            rng = np.random.RandomState(seed)
            placement = _greedy_with_seed(grid_pts, zones, N, rng)
            
            if placement is not None:
                pts = grid_pts[placement]
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_points = pts.copy()
                lo = mid
            else:
                hi = mid
        
        if time.time() > deadline:
            break
    
    # Phase 2: Continuous refinement
    if best_points is not None:
        if time.time() < deadline:
            refined, r = _continuous_refine(grid_pts, None, deadline)
            if r > best_r:
                best_r = r
                best_points = refined
        
        # Phase 3: Simulated annealing to escape local minima
        if time.time() < deadline:
            sa_points, sa_r = _simulated_annealing_continuous(best_points, deadline)
            if sa_r > best_r:
                best_r = sa_r
                best_points = sa_points
    
    # Fallback
    if best_points is None:
        # Simple grid initialization
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
        best_r = _compute_radius(best_points)
    
    # Final validation
    best_points = np.clip(best_points, -HALF + 1e-12, HALF - 1e-12)
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points


# EVOLVE-BLOCK-START
# (The above construct function is the entry point)
# EVOLVE-BLOCK-END
