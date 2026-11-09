"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(circles):
    """
    Compute the score of a set of circles.
    Returns (score, is_valid).
    Score is sum of radii if valid, 0 otherwise.
    """
    if len(circles) == 0:
        return 0.0, True
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check boundaries
    if np.any(r < 0) or np.any(x < 0) or np.any(y < 0) or np.any(x + r > 1.0) or np.any(y + r > 1.0):
        return 0.0, False
        
    # Check overlaps
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _initial_placement():
    """
    Create a dense initial placement.
    We use a hexagonal grid pattern but we will optimize it.
    To ensure we fit 26 circles, we start with a grid that is slightly smaller
    than the theoretical max to allow room for optimization.
    """
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # Area of square is 1. Total area of circles = sum(pi*r^2).
    # Max sum of radii is roughly related to sqrt(area).
    # Let's start with a grid of 5x5 = 25 circles plus 1 extra, or 6x5=30 and remove some?
    # 26 circles. A 5x5 grid has 25. Let's use a 5x5 grid and add one in the middle of a gap?
    # Or just use a 6x5 grid and randomly remove 4?
    # Better: Use a grid of 5x6 = 30 positions, pick the best 26?
    # Actually, let's just generate a 6x6 grid of small circles and let the optimizer expand them.
    # But 36 circles is too many.
    
    # Let's use a 5x5 grid (25 circles) and add one more? 
    # Or a 6x5 grid (30 circles) and remove 4 smallest?
    # Let's try a 5x5 grid first. It's a good baseline.
    # But 26 circles... 5x5 is 25.
    # Let's use a 6x5 grid (30 circles) and keep the first 26?
    # The grid will be dense.
    
    # Strategy: Generate a 6x5 grid of circles.
    # Spacing: 1/6 = 0.166. Radius ~ 0.08.
    # Sum of radii ~ 26 * 0.08 = 2.08.
    # Seed score: 5x5 grid (25 circles) + 1? 
    # Seed code: side = ceil(sqrt(26)) = 6. 
    # Seed uses 6x6 grid, takes first 26.
    # Seed radius: 0.5/6 - 1e-3 = 0.0833 - 0.001 = 0.0823.
    # Seed sum: 26 * 0.0823 = 2.14.
    
    # We need to beat 2.14.
    # Hexagonal packing allows larger radii for same number of circles.
    
    # Let's initialize with a hexagonal pattern.
    # Rows: 5. Cols: 6. Total 30. We need 26.
    # Let's place 26 circles in a 5x6 hexagonal grid, removing 4 corners or edges to allow expansion.
    
    circles = []
    # Hexagonal packing parameters
    # Let r be the radius.
    # Horizontal distance: 2r
    # Vertical distance: sqrt(3)r
    # We want to fit in [0,1]x[0,1].
    # Let's just start with a regular grid of 26 circles and let the optimizer move them.
    # A grid is a safe start. The optimizer will push them into a hex-like arrangement.
    
    # Grid 5x6 = 30. Let's use 5x5=25 and 1 extra?
    # Let's use a 6x5 grid (6 rows, 5 cols) -> 30 circles.
    # We will initialize 26 circles in a 6x5 grid pattern, skipping the last 4.
    
    rows = 6
    cols = 5
    dx = 1.0 / cols
    dy = 1.0 / rows
    
    # Initial positions
    positions = []
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) * dx
            y = (i + 0.5) * dy
            positions.append([x, y])
            
    # Take first 26
    positions = positions[:26]
    
    # Initial radii: slightly less than half the min distance
    # Min distance in grid is min(dx, dy) = 0.2 (for 5 cols) or 0.166 (for 6 rows).
    # Let's set r = 0.08
    r0 = 0.08
    circles = np.array([[p[0], p[1], r0] for p in positions])
    
    return circles

def _optimize(circles, max_time=100.0):
    """
    Alternating minimization:
    1. Fix positions, maximize radii (subject to non-overlap and boundaries).
    2. Fix radii, minimize energy (overlap penalty) to adjust positions.
    """
    start_time = time.time()
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    best_score = 0.0
    best_circles = circles.copy()
    
    # Precompute indices for pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    # Learning rates
    lr_r = 0.005
    lr_p = 0.005
    
    steps = 0
    while time.time() - start_time < max_time:
        steps += 1
        
        # --- Phase 1: Optimize Radii ---
        # For each circle, r_i <= min( x_i, y_i, 1-x_i, 1-y_i, (d_ij - r_j) for all j )
        # This is a linear programming problem, but we can do a simple iterative update.
        # r_i = min(boundary_limit_i, min_j (d_ij - r_j))
        # To avoid oscillation, we can update r_i = 0.5 * (r_i + new_r_i)
        
        new_r = r.copy()
        for i in range(n):
            # Boundary limit
            lim = min(x[i], y[i], 1.0 - x[i], 1.0 - y[i])
            
            # Overlap limits
            # d_ij - r_j must be >= r_i
            # So r_i <= d_ij - r_j
            # We compute the min of these for all j
            # Vectorized:
            dists = np.sqrt((x[i] - x)**2 + (y[i] - y)**2)
            # dists[i] is 0, so exclude self
            # We need min(dists[j] - r[j]) for j != i
            # Note: if dists[j] < r[j] + r[i], they overlap.
            # The constraint is r_i + r_j <= dists[i,j]
            # So r_i <= dists[i,j] - r_j
            
            # Create a mask to exclude i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            candidate_limits = dists[mask] - r[mask]
            min_overlap_lim = np.min(candidate_limits)
            
            new_r[i] = min(lim, min_overlap_lim)
            
            # Ensure non-negative
            if new_r[i] < 0:
                new_r[i] = 0.0
                
        # Update with damping
        r = 0.5 * r + 0.5 * new_r
        
        # --- Phase 2: Optimize Positions ---
        # Repulsion forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        for i in range(n):
            # Boundary forces
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 2.0
            if x[i] > 1.0 - r[i]:
                fx[i] += (1.0 - r[i] - x[i]) * 2.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 2.0
            if y[i] > 1.0 - r[i]:
                fy[i] += (1.0 - r[i] - y[i]) * 2.0
            
        # Circle-Circle Repulsion
        # For each pair, if dist < r_i + r_j, push apart
        # Force magnitude proportional to overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    # Random nudge
                    dx = np.random.uniform(-0.01, 0.01)
                    dy = np.random.uniform(-0.01, 0.01)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                overlap = (r[i] + r[j]) - dist
                if overlap > 0:
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    # Force is proportional to overlap
                    f = overlap * 1.0
                    fx[i] += nx * f
                    fy[i] += ny * f
                    fx[j] -= nx * f
                    fy[j] -= ny * f
        
        # Update positions
        x += lr_p * fx
        y += lr_p * fy
        
        # Clamp to boundaries (soft)
        # Actually, the force should handle it, but let's ensure we don't go too far
        # We don't hard clamp because that can cause sticking, but we can limit the step size
        # The force is linear, so it should be stable with small lr.
        
        # Check score
        current_circles = np.column_stack((x, y, r))
        score, is_valid = _compute_score(current_circles)
        if is_valid and score > best_score:
            best_score = score
            best_circles = current_circles.copy()
            
        # Adaptive learning rate?
        # If score improves, keep lr. If not, reduce?
        # For simplicity, keep fixed small lr.
        
    return best_circles, best_score

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # 1. Initial placement
    circles = _initial_placement()
    
    # 2. Optimization
    optimized_circles, score = _optimize(circles, max_time=deadline - (time.time() - start_time))
    
    # 3. Final validation and return
    # The optimizer returns the best valid configuration found.
    # We should double check it's valid.
    final_score, is_valid = _compute_score(optimized_circles)
    if not is_valid:
        # Fallback to initial grid if optimization failed completely (shouldn't happen with damping)
        # Or return the best valid one found internally.
        # _optimize tracks best_valid.
        # If _optimize returns invalid, we have a bug. 
        # Let's assume _optimize returns valid.
        pass
        
    return optimized_circles
# EVOLVE-BLOCK-END
