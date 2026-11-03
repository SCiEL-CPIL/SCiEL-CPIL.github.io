"""Pack 200 equal circles in a unit square using greedy lattice selection with local refinement."""
import numpy as np
import time

N = 200
_TIME_LIMIT = 115  # seconds, leave margin before 120s kill

def _compute_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    
    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    
    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)
    
    # Min pairwise distance
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair_dist = np.min(D)
    
    return min(wall_dist, min_pair_dist / 2.0)

def _generate_hex_grid(spacing, square_size=1.0):
    """Generate a hexagonal grid of points within the unit square."""
    points = []
    # Hex grid: rows offset by half spacing
    row_height = spacing * np.sqrt(3) / 2
    n_rows = int(square_size / row_height) + 2
    n_cols = int(square_size / spacing) + 2
    
    for i in range(n_rows):
        y = -0.5 + (i + 0.5) * row_height
        if y < -0.5 or y > 0.5:
            continue
        offset = spacing / 2 if i % 2 == 1 else 0.0
        for j in range(n_cols):
            x = -0.5 + (j + 0.5) * spacing + offset
            if x >= -0.5 and x <= 0.5:
                points.append((x, y))
    
    return np.array(points) if points else np.empty((0, 2))

def _greedy_select(candidate_points, N, target_radius=None):
    """
    Greedy selection: pick N points from candidate_points to maximize
    the minimum distance to already selected points and walls.
    """
    if len(candidate_points) < N:
        return candidate_points.copy()
    
    selected = []
    remaining = list(range(len(candidate_points)))
    
    # Precompute wall distances for each candidate
    wall_dists = np.zeros(len(candidate_points))
    for i, pt in enumerate(candidate_points):
        wall_dists[i] = min(pt[0] + 0.5, 0.5 - pt[0], pt[1] + 0.5, 0.5 - pt[1])
    
    while len(selected) < N and remaining:
        best_idx = -1
        best_score = -1
        best_min_dist = 0
        
        for idx in remaining:
            pt = candidate_points[idx]
            
            # Score: min distance to walls and to all selected points
            min_d = wall_dists[idx]
            
            if selected:
                sel_pts = candidate_points[selected]
                dists = np.linalg.norm(pt - sel_pts, axis=1)
                min_d = min(min_d, np.min(dists) / 2.0)
            
            # We want to maximize the minimum distance
            if min_d > best_score:
                best_score = min_d
                best_idx = idx
                best_min_dist = min_d
        
        if best_idx == -1:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    return candidate_points[selected]

def _local_refinement(pts, time_limit=30.0):
    """
    Local refinement using a simple force-directed approach.
    """
    t0 = time.time()
    pts = pts.copy()
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    T = 0.005
    cooling = 0.9995
    min_T = 1e-7
    
    k_repel = 50.0
    k_wall = 20.0
    
    while time.time() - t0 < time_limit:
        forces = np.zeros_like(pts)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # Wall distances
        dist_left = pts[:, 0] + 0.5
        dist_right = 0.5 - pts[:, 0]
        dist_bottom = pts[:, 1] + 0.5
        dist_top = 0.5 - pts[:, 1]
        wall_dists = np.minimum.reduce([dist_left, dist_right, dist_bottom, dist_top])
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = pts.copy()
        
        # Repulsion between close pairs
        threshold = max(min_pair_dist * 1.1, 0.001)
        mask = dists < threshold
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vecs = diff / np.where(dists > 0, dists[:, :, np.newaxis], 1.0)
        unit_vecs[np.eye(len(pts), dtype=bool)] = 0
        mag = np.where(mask, k_repel * (threshold - dists), 0.0)
        forces += np.sum(mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall repulsion
        margin = current_r + 0.0005
        close_left = pts[:, 0] < -0.5 + margin
        forces[close_left, 0] += k_wall * (margin - (pts[close_left, 0] + 0.5))
        
        close_right = pts[:, 0] > 0.5 - margin
        forces[close_right, 0] -= k_wall * (margin - (0.5 - pts[close_right, 0]))
        
        close_bottom = pts[:, 1] < -0.5 + margin
        forces[close_bottom, 1] += k_wall * (margin - (pts[close_bottom, 1] + 0.5))
        
        close_top = pts[:, 1] > 0.5 - margin
        forces[close_top, 1] -= k_wall * (margin - (0.5 - pts[close_top, 1]))
        
        # Small centering force
        forces += -0.1 * pts
        
        # Add noise
        noise = np.random.normal(0, T, size=pts.shape)
        steps = forces + noise
        
        # Limit step size
        max_step = 0.02
        step_norms = np.linalg.norm(steps, axis=1)
        for i in range(len(pts)):
            if step_norms[i] > max_step:
                steps[i] = steps[i] / step_norms[i] * max_step
        
        pts += steps
        
        # Clamp to square
        pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
        pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
        
        # Cool
        T *= cooling
        if T < min_T:
            T = min_T
    
    return best_pts

def construct():
    t0 = time.time()
    
    # Strategy: Try multiple hex grid spacings, do greedy selection, then refine.
    best_pts = None
    best_r = 0.0
    
    # Try a range of hex grid spacings
    # Expected optimal spacing is around 2*r ~ 0.073
    # Try spacings from 0.05 to 0.10
    spacings = np.linspace(0.05, 0.10, 11)
    
    for spacing in spacings:
        if time.time() - t0 > _TIME_LIMIT - 15:
            break
        
        # Generate hex grid
        candidates = _generate_hex_grid(spacing)
        
        if len(candidates) < N:
            continue
        
        # Greedy selection
        selected = _greedy_select(candidates, N)
        
        if len(selected) < N:
            continue
        
        # Compute radius
        r = _compute_radius(selected)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
        
        # Local refinement with time budget
        remaining_time = _TIME_LIMIT - (time.time() - t0)
        if remaining_time > 5:
            refined = _local_refinement(selected, time_limit=min(remaining_time - 2, 10))
            r_refined = _compute_radius(refined)
            
            if r_refined > best_r:
                best_r = r_refined
                best_pts = refined.copy()
    
    # If we didn't get a good result, try a rectangular grid as fallback
    if best_r < 0.03:
        rows, cols = 20, 10
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        best_r = _compute_radius(best_pts)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
