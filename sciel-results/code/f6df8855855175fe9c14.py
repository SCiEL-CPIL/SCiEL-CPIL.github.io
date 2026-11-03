"""
Pack 200 equal circles in a unit square using coordinate ascent with targeted
bottleneck moves and multiple diverse starts.
"""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
_TIME_LIMIT = 140  # Leave margin for 150s kill

def _compute_radius_fast(pts):
    """
    Fast radius computation using cdist.
    """
    if len(pts) == 0 or not np.all(np.isfinite(pts)):
        return 0.0
    
    # Wall distances
    min_wall = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    
    if min_wall < 0:
        return 0.0

    # Pairwise distances
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _local_optimize(coords, max_iter=500, time_limit=None, start_time=None):
    """
    Coordinate ascent with targeted bottleneck moves.
    At each iteration, find the closest pair and wall-violating points,
    then move them to increase the minimum distance.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = _TIME_LIMIT - 10
    
    coords = coords.copy()
    n = len(coords)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute pairwise distances
        D = cdist(coords, coords)
        np.fill_diagonal(D, np.inf)
        
        # Find closest pair
        min_dist = np.min(D)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        
        # Wall distances for each point
        wall_dists = np.array([
            coords[:, 0] + 0.5,
            0.5 - coords[:, 0],
            coords[:, 1] + 0.5,
            0.5 - coords[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # Current radius
        r = min(min_dist / 2.0, min_wall)
        
        # If no improvement possible, break
        if r <= 0:
            break
        
        # Strategy: move the bottleneck points apart
        # Try moving i and j apart
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 1e-12:
            # Move i in direction away from j, j in direction away from i
            move = 0.01 * r
            nx, ny = dx / dist, dy / dist
            
            # Try moving i
            new_i = coords[i].copy()
            new_i[0] += nx * move
            new_i[1] += ny * move
            new_i = np.clip(new_i, -0.5, 0.5)
            
            # Check if this improves
            coords[i] = new_i
            new_r = _compute_radius_fast(coords)
            if new_r <= r:
                coords[i] = new_i - np.array([nx * move, ny * move])
            
            # Try moving j
            new_j = coords[j].copy()
            new_j[0] -= nx * move
            new_j[1] -= ny * move
            new_j = np.clip(new_j, -0.5, 0.5)
            
            coords[j] = new_j
            new_r2 = _compute_radius_fast(coords)
            if new_r2 <= r:
                coords[j] = new_j + np.array([nx * move, ny * move])
        
        # Also try moving wall-violating points inward
        for axis in range(2):
            for side in range(2):
                # side 0: left/bottom, side 1: right/top
                if side == 0:
                    wall_dist = coords[:, axis] + 0.5
                else:
                    wall_dist = 0.5 - coords[:, axis]
                
                min_idx = np.argmin(wall_dist)
                if wall_dist[min_idx] < r:
                    # Move this point inward
                    if side == 0:
                        coords[min_idx, axis] += 0.01 * r
                    else:
                        coords[min_idx, axis] -= 0.01 * r
                    coords[min_idx, axis] = np.clip(coords[min_idx, axis], -0.5, 0.5)
        
        # Small random perturbation to escape local minima
        if it % 10 == 0 and it > 0:
            perturb = 0.005 * r * np.random.randn(n, 2)
            coords += perturb
            coords = np.clip(coords, -0.5, 0.5)
    
    return coords

def _hexagonal_initialization(n, jitter=0.0):
    """
    Generate a hexagonal layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    r_est = 0.5 / m
    
    cols = int(1.0 / (2 * r_est)) + 2
    rows = int(1.0 / (r_est * np.sqrt(3))) + 2
    
    points = []
    for row in range(rows):
        y = -0.5 + r_est + row * r_est * np.sqrt(3)
        offset = 0.0 if row % 2 == 0 else r_est
        for col in range(cols):
            x = -0.5 + r_est + offset + col * 2 * r_est
            points.append([x, y])
            
    points = np.array(points)
    
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) < n:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:n]
    else:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)
        points = points[indices[:n]]
    
    if jitter > 0:
        points += jitter * np.random.randn(n, 2)
        points = np.clip(points, -0.5, 0.5)
    
    return points

def _grid_initialization(n, jitter=0.0):
    """
    Generate a grid layout with optional jitter.
    """
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    if len(grid) < n:
        # Add more points
        extra = n - len(grid)
        extra_pts = np.random.uniform(-0.45, 0.45, (extra, 2))
        grid = np.vstack([grid, extra_pts])
    
    grid = grid[:n]
    
    if jitter > 0:
        grid += jitter * np.random.randn(n, 2)
        grid = np.clip(grid, -0.5, 0.5)
    
    return grid

def _random_initialization(n):
    """
    Generate random points.
    """
    return np.random.uniform(-0.45, 0.45, (n, 2))

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Multiple diverse starts
    initializers = [
        ("hex", lambda: _hexagonal_initialization(N, jitter=0.0)),
        ("hex_jitter", lambda: _hexagonal_initialization(N, jitter=0.02)),
        ("grid", lambda: _grid_initialization(N, jitter=0.0)),
        ("grid_jitter", lambda: _grid_initialization(N, jitter=0.02)),
        ("random", lambda: _random_initialization(N)),
    ]
    
    best_coords = None
    best_radius = 0.0
    
    for name, init_func in initializers:
        if time.time() > deadline - 20:
            break
        
        try:
            coords = init_func()
            r = _compute_radius_fast(coords)
            
            if r > best_radius:
                best_radius = r
                best_coords = coords.copy()
            
            # Local optimization
            remaining_time = deadline - time.time() - 5
            if remaining_time > 5:
                optimized = _local_optimize(coords, max_iter=300, 
                                           time_limit=remaining_time,
                                           start_time=start_time)
                r_opt = _compute_radius_fast(optimized)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_coords = optimized.copy()
        except Exception:
            continue
    
    # Final validation
    if best_coords is None or not np.all(np.isfinite(best_coords)):
        return _grid_initialization(N)
    
    return best_coords

# EVOLVE-BLOCK-START
def compute_radius(coords):
    return _compute_radius_fast(coords)

def objective(coords):
    r = compute_radius(coords)
    return -r

def initial_hex_layout(n):
    return _hexagonal_initialization(n)

# EVOLVE-BLOCK-END
