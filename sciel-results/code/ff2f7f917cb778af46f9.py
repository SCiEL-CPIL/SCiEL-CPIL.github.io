import numpy as np
import time
import sys
from scipy.spatial.distance import cdist

N = 200
TIME_LIMIT = 145.0  # Seconds, leaving margin before 150s kill
START_TIME = time.time()

def _check_time():
    return time.time() - START_TIME

def _compute_radius(centres):
    """Compute the packing radius for a set of centres.
    Returns r if valid, 0.0 if invalid.
    """
    if len(centres) != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check bounds: centres must be inside [-0.5, 0.5]^2
    if np.any(centres < -0.5) or np.any(centres > 0.5):
        return 0.0
    
    # Distance to walls
    dx = np.abs(centres[:, 0])
    dy = np.abs(centres[:, 1])
    wall_dists = np.min(np.stack([0.5 - dx, 0.5 - dy], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0

    # Minimum pairwise distance
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)

    return min(min_dist / 2.0, min_wall)

def _generate_initial_config():
    """Generate a high-quality initial configuration using a hybrid grid/hex approach."""
    best_centres = None
    best_r = -1
    
    # Strategy 1: Hexagonal Packing with optimal spacing search
    # For hexagonal packing, the density is pi/sqrt(12) ~ 0.9069
    # For N=200, area = 200 * pi * r^2 = 1 (unit square area)
    # r^2 = 1 / (200 * pi) => r ~ 0.03989
    # But we are constrained by the square boundary.
    
    # Let's construct a hex grid and find the best spacing that fits ~200 points.
    # Hex grid points:
    # Row j: y = j * h, where h = s * sqrt(3)/2
    # Col i: x = i * s + (j % 2) * s/2
    
    # We want to fit as many as possible, or exactly N by trimming/adjusting.
    # Since we need exactly N, we'll generate a slightly larger hex grid and pick the best N?
    # No, that's hard. Better: Adjust the grid dimensions to fit exactly N?
    # Hex grids don't have exact rectangular counts easily.
    
    # Alternative: Start with a square grid 14x14 = 196, add 4? No.
    # 10x20 = 200. This is the only "clean" rectangular grid.
    # Let's optimize the 10x20 grid with hexagonal distortion.
    
    # Also try 11x18 = 198 (too few), 12x17 = 204 (too many).
    # So 10x20 is the target topology.
    
    # Generate 20x10 grid (20 cols, 10 rows)
    w, h = 20, 10
    # Spacing s_x and s_y
    # To maximize radius, we want to fill the square [-0.5, 0.5]
    # s_x = 1.0 / (w - 1)
    # s_y = 1.0 / (h - 1)
    # But hexagonal packing implies s_y = s_x * sqrt(3)/2
    # If we force hexagonal geometry, we might not fill the square perfectly.
    # Let's try to fit a hex pattern into the square.
    
    # Approach:
    # 1. Start with a 20x10 grid.
    # 2. Apply hexagonal offsets.
    # 3. Optimize positions.
    
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    
    # Scale to fit in [-0.5, 0.5]
    # If we just use unit spacing, the width is w-1 and height is h-1.
    # We want width <= 1 and height <= 1.
    # Let's scale to fit width 1.
    scale = 1.0 / (w - 1)
    grid = np.array([(x * scale, y * scale) for y in ys for x in xs])
    
    # Now apply hexagonal distortion.
    # In a hex grid, odd rows are shifted by s/2.
    # Here s = scale.
    for i in range(h):
        if i % 2 != 0:
            grid[i*w:(i+1)*w, 0] += scale / 2.0
            
    # This might push points outside. Let's check and clip?
    # Clipping breaks the packing.
    # Instead, we should center the grid and ensure it fits.
    
    # Let's try a different initialization:
    # Generate a hex grid that fits in the square.
    
    best_r = -1
    best_centres = None
    
    # Try different hex spacings
    for s in np.linspace(0.04, 0.05, 20):
        h_hex = s * np.sqrt(3) / 2.0
        points = []
        # Determine number of rows
        max_y = 0.5
        num_rows = int(max_y / h_hex) + 1
        for j in range(-num_rows, num_rows + 1):
            y = j * h_hex
            if abs(y) > 0.5:
                continue
            offset = (j % 2) * s / 2.0
            # Determine number of cols
            max_x = 0.5
            num_cols = int(max_x / s) + 1
            for i in range(-num_cols, num_cols + 1):
                x = i * s + offset
                if -0.5 <= x <= 0.5:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # We have more than N points. We need to select N.
        # Which ones to remove?
        # Ideally, remove the ones that are least "central" or cause the most conflict?
        # Actually, if we have a valid hex grid with > N points, we can just remove points.
        # But removing points increases the available space, so the radius can increase.
        # However, the remaining points are still constrained by their neighbors.
        # If we remove points, the min distance among remaining points stays the same (s).
        # The wall distance might improve if we remove edge points.
        
        # Let's just take the first N points? No, that's biased.
        # Let's take the points closest to the center?
        # Or, better: if we have a hex grid with spacing s, the radius is limited by s/2 and wall.
        # If we have > N points, we can increase s until we have exactly N points?
        # This is complex.
        
        # Simpler: Use the 20x10 grid with hex distortion, then optimize.
        pass
    
    # Let's stick to the 20x10 grid with hex distortion as the primary seed.
    # It's a valid topology for N=200.
    
    w, h = 20, 10
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    
    # Scale to fit width 1
    scale = 1.0 / (w - 1)
    grid = np.array([(x * scale, y * scale) for y in ys for x in xs])
    
    # Apply hex offset to odd rows
    for i in range(h):
        if i % 2 != 0:
            grid[i*w:(i+1)*w, 0] += scale / 2.0
            
    # The grid might be slightly wider than 1.0 due to offset.
    # Let's check bounds.
    # If out of bounds, scale down.
    max_x = np.max(np.abs(grid[:, 0]))
    max_y = np.max(np.abs(grid[:, 1]))
    if max_x > 0.5 or max_y > 0.5:
        scale_down = 0.5 / max(max_x, max_y)
        grid *= scale_down
        
    r = _compute_radius(grid)
    if r > best_r:
        best_r = r
        best_centres = grid
        
    # Also try a pure square grid for comparison
    xs = (np.arange(w) - (w-1)/2.0)
    ys = (np.arange(h) - (h-1)/2.0)
    scale = 1.0 / (w - 1)
    grid_sq = np.array([(x * scale, y * scale) for y in ys for x in xs])
    r_sq = _compute_radius(grid_sq)
    if r_sq > best_r:
        best_r = r_sq
        best_centres = grid_sq
        
    return best_centres

def _local_optimize(centres, time_limit):
    """
    Optimize the positions to maximize the minimum radius.
    Uses a coordinate descent / gradient-free approach with momentum.
    """
    current_r = _compute_radius(centres)
    if current_r == 0.0:
        return centres, 0.0
        
    best_centres = centres.copy()
    best_r = current_r
    
    deadline = START_TIME + time_limit
    step_size = 0.005
    min_step_size = 1e-6
    
    # We will use a "push" strategy.
    # Identify the bottleneck pair or wall.
    # Move points to increase the minimum distance.
    
    while time.time() < deadline and step_size > min_step_size:
        # Calculate constraints
        wall_clearances = np.min(np.stack([
            0.5 - np.abs(best_centres[:, 0]),
            0.5 - np.abs(best_centres[:, 1])
        ], axis=1), axis=1)
        
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dists = np.min(dists, axis=1)
        
        clearances = np.minimum(min_neighbor_dists / 2.0, wall_clearances)
        bottleneck_idx = np.argmin(clearances)
        bottleneck_val = clearances[bottleneck_idx]
        
        if bottleneck_val > best_r + 1e-9:
            # We improved? This shouldn't happen if we only accept improvements.
            # But maybe numerical noise.
            pass
            
        # Determine move direction for bottleneck point
        cx, cy = best_centres[bottleneck_idx]
        
        # Check wall
        dist_left = cx + 0.5
        dist_right = 0.5 - cx
        dist_down = cy + 0.5
        dist_up = 0.5 - cy
        min_wall_dist = min(dist_left, dist_right, dist_down, dist_up)
        
        # Check neighbor
        neighbor_idx = np.argmin(dists[bottleneck_idx])
        neighbor_dist = dists[bottleneck_idx, neighbor_idx]
        
        if neighbor_dist < 2.0 * min_wall_dist:
            # Neighbor is the constraint
            nx, ny = best_centres[neighbor_idx]
            dx = cx - nx
            dy = cy - ny
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 1e-9:
                dx /= norm
                dy /= norm
            else:
                dx, dy = 1.0, 0.0
            move_x, move_y = dx * step_size, dy * step_size
        else:
            # Wall is the constraint
            if min_wall_dist == dist_left:
                move_x, move_y = step_size, 0.0
            elif min_wall_dist == dist_right:
                move_x, move_y = -step_size, 0.0
            elif min_wall_dist == dist_down:
                move_x, move_y = 0.0, step_size
            else:
                move_x, move_y = 0.0, -step_size
        
        new_centres = best_centres.copy()
        new_centres[bottleneck_idx] += [move_x, move_y]
        new_centres[bottleneck_idx] = np.clip(new_centres[bottleneck_idx], -0.5, 0.5)
        
        new_r = _compute_radius(new_centres)
        
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centres = new_centres
            step_size *= 0.98 # Shrink slowly
        else:
            step_size *= 0.9
            if step_size < 1e-4:
                # Random kick
                kick_idx = np.random.randint(N)
                kick_x = np.random.uniform(-step_size*10, step_size*10)
                kick_y = np.random.uniform(-step_size*10, step_size*10)
                new_centres = best_centres.copy()
                new_centres[kick_idx] += [kick_x, kick_y]
                new_centres[kick_idx] = np.clip(new_centres[kick_idx], -0.5, 0.5)
                new_r = _compute_radius(new_centres)
                if new_r > best_r + 1e-12:
                    best_r = new_r
                    best_centres = new_centres
                    step_size = 0.005
                else:
                    step_size *= 0.8
    
    return best_centres, best_r

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - (start_time - START_TIME)
    
    # 1. Generate Initial Configuration
    centres = _generate_initial_config()
    current_r = _compute_radius(centres)
    
    if current_r == 0.0:
        # Emergency fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # 2. Local Optimization
    if time_budget > 10:
        centres, current_r = _local_optimize(centres, time_budget - 5)
        
    # 3. Final Validation
    final_r = _compute_radius(centres)
    if final_r <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    return centres
