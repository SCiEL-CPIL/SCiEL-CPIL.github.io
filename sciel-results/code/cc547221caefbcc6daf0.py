"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200
_START_TIME = time.time()
_TIME_LIMIT = 110  # seconds, leave margin before 120s kill

def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
        
    return min(min_pair / 2.0, min_boundary)

def _hex_grid_points(params):
    """
    Generate points for a hexagonal grid.
    params: [dx, dy, x0, y0, offset]
    dx: horizontal spacing
    dy: vertical spacing (should be dx * sqrt(3)/2 for perfect hex)
    x0, y0: origin offset
    offset: stagger amount for odd rows (0 to 1, 0.5 is perfect hex stagger)
    """
    dx, dy, x0, y0, off = params
    if dx <= 1e-6 or dy <= 1e-6:
        return None
    
    # Determine number of rows and cols to fit ~N points
    # We iterate over a few likely configurations in the caller, 
    # but here we just generate a grid large enough and truncate.
    # To be efficient, we estimate rows/cols from the square aspect ratio.
    # For N=200, approx 14x14.
    
    # We will generate points row by row until we have N
    pts = []
    # Estimate max rows/cols to avoid infinite loop
    max_rows = 30
    max_cols = 30
    
    for r in range(max_rows):
        y = y0 + r * dy
        if y > 0.5:
            break
        shift = off * dx if (r % 2 == 1) else 0.0
        for c in range(max_cols):
            x = x0 + c * dx + shift
            if x > 0.5:
                break
            pts.append((x, y))
            if len(pts) >= N:
                return np.array(pts)
    return None

def _rect_grid_points(params):
    """
    Generate points for a rectangular grid.
    params: [dx, dy, x0, y0]
    """
    dx, dy, x0, y0 = params
    if dx <= 1e-6 or dy <= 1e-6:
        return None
        
    pts = []
    max_rows = 30
    max_cols = 30
    
    for r in range(max_rows):
        y = y0 + r * dy
        if y > 0.5:
            break
        for c in range(max_cols):
            x = x0 + c * dx
            if x > 0.5:
                break
            pts.append((x, y))
            if len(pts) >= N:
                return np.array(pts)
    return None

def _score_hex(params):
    pts = _hex_grid_points(params)
    if pts is None or len(pts) != N:
        return 0.0
    return _compute_radius(pts)

def _score_rect(params):
    pts = _rect_grid_points(params)
    if pts is None or len(pts) != N:
        return 0.0
    return _compute_radius(pts)

def _neg_score_hex(params):
    return -_score_hex(params)

def _neg_score_rect(params):
    return -_score_rect(params)

def _generate_initial_candidates():
    """
    Generate a set of high-quality initial candidates by optimizing
    lattice parameters.
    """
    candidates = []
    
    # 1. Hexagonal Grids
    # Try different aspect ratios / row-col splits
    # For N=200, sqrt(N) ~ 14.
    # We don't need exact factors, just good density.
    
    # Standard hex spacing: dx = 1, dy = sqrt(3)/2 * dx
    # Let's optimize dx, dy, x0, y0, offset
    
    # Initial guess: fit 14 cols and 15 rows roughly
    # Width ~ 1, Height ~ 1
    # dx ~ 1/14, dy ~ 1/15
    
    hex_configs = [
        # [dx, dy, x0, y0, offset]
        # Centered roughly
        (1.0/14, 1.0/15 * np.sqrt(3)/2, -0.5 + 1.0/28, -0.5 + 1.0/30 * np.sqrt(3)/2, 0.5),
        (1.0/15, 1.0/14 * np.sqrt(3)/2, -0.5 + 1.0/30, -0.5 + 1.0/28 * np.sqrt(3)/2, 0.5),
        (1.0/13, 1.0/16 * np.sqrt(3)/2, -0.5 + 1.0/26, -0.5 + 1.0/32 * np.sqrt(3)/2, 0.5),
        (1.0/16, 1.0/13 * np.sqrt(3)/2, -0.5 + 1.0/32, -0.5 + 1.0/26 * np.sqrt(3)/2, 0.5),
    ]
    
    for config in hex_configs:
        x0 = np.array(config)
        try:
            res = minimize(_neg_score_hex, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            score = -res.fun
            if score > 0:
                pts = _hex_grid_points(res.x)
                if pts is not None and len(pts) == N:
                    candidates.append((score, pts))
        except Exception:
            pass
            
    # 2. Rectangular Grids
    rect_configs = [
        # [dx, dy, x0, y0]
        (1.0/14, 1.0/15, -0.5 + 1.0/28, -0.5 + 1.0/30),
        (1.0/15, 1.0/14, -0.5 + 1.0/30, -0.5 + 1.0/28),
        (1.0/10, 1.0/20, -0.5 + 1.0/20, -0.5 + 1.0/40),
        (1.0/20, 1.0/10, -0.5 + 1.0/40, -0.5 + 1.0/20),
    ]
    
    for config in rect_configs:
        x0 = np.array(config)
        try:
            res = minimize(_neg_score_rect, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            score = -res.fun
            if score > 0:
                pts = _rect_grid_points(res.x)
                if pts is not None and len(pts) == N:
                    candidates.append((score, pts))
        except Exception:
            pass

    if not candidates:
        # Fallback to simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N]
        r = _compute_radius(pts)
        candidates.append((r, pts))
        
    return candidates

def _bottleneck_refine(points, time_budget=50.0):
    """
    Refine the packing by specifically moving points that are part of the
    minimum distance pair or closest to the boundary.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    step = 0.001
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        # Identify bottlenecks
        # 1. Closest pair
        # 2. Closest to boundary
        
        # Compute pairwise distances
        d = pdist(points)
        min_pair_dist = np.min(d)
        min_pair_idx = np.argmin(d)
        
        # Convert condensed index to (i, j)
        # scipy.spatial.distance.pdist uses condensed form
        # We need to find i, j such that i < j and index matches
        # This is O(N^2) to map, but N=200 is small. 
        # Alternatively, we can just iterate or use a lookup.
        # For speed, we can just check all pairs if N is small, but pdist is fast.
        # Let's just find the indices manually for clarity, it's fast enough for N=200.
        
        # Actually, let's just recompute min dist and indices in a vectorized way if possible,
        # or stick to the condensed index mapping.
        
        # Mapping condensed index to (i, j):
        # index = i * (N - 1) - i * (i - 1) / 2 + (j - i - 1)
        # It's easier to just loop for N=200.
        
        min_i, min_j = -1, -1
        min_d_val = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_d_val:
                    min_d_val = dist
                    min_i, min_j = i, j
        
        # Boundary distances
        dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
        min_boundary_idx = np.argmin(dist_boundary)
        min_boundary_val = dist_boundary[min_boundary_idx]
        
        # Determine which constraint is tighter
        # Radius is min(min_pair_dist / 2, min_boundary_val)
        
        improved = False
        
        # Try to move the bottleneck points
        # If pair is bottleneck, move them apart
        if min_pair_dist / 2.0 <= min_boundary_val:
            # Move min_i and min_j apart
            diff = points[min_i] - points[min_j]
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-9:
                dir_i = diff / norm_diff
                dir_j = -diff / norm_diff
            else:
                dir_i = np.array([1.0, 0.0])
                dir_j = np.array([-1.0, 0.0])
            
            # Try small moves
            for s in [step, step * 0.5, step * 0.2]:
                new_pts = points.copy()
                new_pts[min_i] += dir_i * s
                new_pts[min_j] += dir_j * s
                
                # Check bounds
                if np.all((new_pts >= -0.5) & (new_pts <= 0.5)):
                    new_r = _compute_radius(new_pts)
                    if new_r > current_r + 1e-12:
                        points = new_pts
                        current_r = new_r
                        improved = True
                        no_improve_count = 0
                        break
        else:
            # Boundary is bottleneck
            idx = min_boundary_idx
            # Move away from the closest wall
            # Determine which wall is closest
            x, y = points[idx]
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bottom = y + 0.5
            dist_top = 0.5 - y
            
            min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
            
            if min_dist == dist_left:
                dir = np.array([1.0, 0.0])
            elif min_dist == dist_right:
                dir = np.array([-1.0, 0.0])
            elif min_dist == dist_bottom:
                dir = np.array([0.0, 1.0])
            else:
                dir = np.array([0.0, -1.0])
            
            for s in [step, step * 0.5, step * 0.2]:
                new_pts = points.copy()
                new_pts[idx] += dir * s
                
                if np.all((new_pts >= -0.5) & (new_pts <= 0.5)):
                    new_r = _compute_radius(new_pts)
                    if new_r > current_r + 1e-12:
                        points = new_pts
                        current_r = new_r
                        improved = True
                        no_improve_count = 0
                        break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 5:
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
                
    return points, current_r

def construct():
    deadline = time.time() + _TIME_LIMIT
    
    # 1. Generate initial candidates
    candidates = _generate_initial_candidates()
    
    # Sort by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_points = candidates[0][1]
    best_r = candidates[0][0]
    
    # 2. Refine the best candidate
    if time.time() < deadline - 5:
        time_budget = min(60.0, deadline - time.time() - 2)
        best_points, best_r = _bottleneck_refine(best_points, time_budget)
        
    # 3. If time remains, try refining the second best candidate?
    # Usually the first one is significantly better after optimization.
    # But if the first one stagnates, maybe the second one has potential.
    # Given the bottleneck refine is fast, we can afford to check.
    
    if len(candidates) > 1 and time.time() < deadline - 10:
        second_points, second_r = candidates[1]
        # Only if it's close to the best
        if second_r > best_r * 0.95:
            time_budget = min(20.0, deadline - time.time() - 2)
            if time_budget > 0:
                refined_pts, refined_r = _bottleneck_refine(second_points, time_budget)
                if refined_r > best_r:
                    best_points = refined_pts
                    best_r = refined_r

    return best_points
