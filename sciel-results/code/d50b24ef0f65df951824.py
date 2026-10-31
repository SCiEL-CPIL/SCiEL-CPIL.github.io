import numpy as np
import time
import math

N = 21
# Deadline guard: The evaluator kills at 150s. We aim to finish well before.
START_TIME = time.time()
DEADLINE = START_TIME + 135.0  # 15s safety margin

def _pairwise_distances(x, y):
    """
    Compute pairwise distance matrix for N points.
    x, y: shape (N,)
    returns: shape (N, N)
    """
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _force_step(x, y, r, w, h, dt=0.05, k_overlap=100.0, k_wall=50.0):
    """
    Perform one step of force-directed relaxation.
    Returns updated x, y.
    """
    n = len(x)
    fx = np.zeros(n)
    fy = np.zeros(n)
    
    # 1. Pairwise repulsion
    d = _pairwise_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-9, d, 1e-9)
    
    # Overlap vector: positive if overlapping
    # We want force proportional to overlap, directed along the line connecting centers
    # F = k * overlap * (dx/d, dy/d)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    
    # Normalize direction
    dir_x = dx / d_safe
    dir_y = dy / d_safe
    
    # Overlap amount (only where d < r_sum)
    overlap = np.maximum(0, r_sum - d)
    
    # Force magnitude
    force_mag = k_overlap * overlap
    
    # Accumulate forces
    # Sum over all j for each i
    fx += np.sum(force_mag * dir_x, axis=1)
    fy += np.sum(force_mag * dir_y, axis=1)
    
    # 2. Wall repulsion
    # Left/Right walls
    # If x - r < 0, push right. If x + r > w, push left.
    dist_left = x - r
    dist_right = w - x - r
    dist_bottom = y - r
    dist_top = h - y - r
    
    # Penalties are linear or quadratic? Linear is stiffer near boundary.
    # Let's use a stiff spring: F = -k * displacement if inside boundary violation
    # Actually, standard penalty: if dist < 0, force = k * |dist|
    
    fx -= k_wall * np.maximum(0, -dist_left)  # Push right if too close to left
    fx += k_wall * np.maximum(0, -dist_right) # Push left if too close to right
    fy -= k_wall * np.maximum(0, -dist_bottom) # Push up if too close to bottom
    fy += k_wall * np.maximum(0, -dist_top)    # Push down if too close to top
    
    # Update positions
    x_new = x + dt * fx
    y_new = y + dt * fy
    
    # Hard clamp to keep them in bounds (prevents explosion)
    # Note: Clamping can cause sticking, but combined with forces it's robust
    x_new = np.clip(x_new, r, w - r)
    y_new = np.clip(y_new, r, h - r)
    
    return x_new, y_new

def _check_validity(x, y, r, w, h, tol=1e-7):
    """
    Check if the configuration is valid.
    Returns (is_valid, sum_radii)
    """
    if w <= 0 or h <= 0:
        return False, 0.0
        
    # Check bounds
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False, 0.0
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False, 0.0
        
    # Check overlaps
    d = _pairwise_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    
    # Check upper triangle
    triu_indices = np.triu_indices(N, k=1)
    if np.any(d[triu_indices] < r_sum[triu_indices] - tol):
        return False, 0.0
        
    return True, float(np.sum(r))

def _grow_radii(x, y, r, w, h, max_iter=1000, growth_factor=0.01):
    """
    Iteratively grow radii while maintaining non-overlap and bounds.
    This is the "polish" step.
    """
    n = len(r)
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
            
        # Calculate max possible growth for each circle
        # Limited by distance to walls and distance to other circles
        growths = np.zeros(n)
        
        for i in range(n):
            # Wall limits
            g_max = min(x[i] - r[i], w - x[i] - r[i], y[i] - r[i], h - y[i] - r[i])
            if g_max < 0:
                g_max = 0
            
            # Neighbor limits
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.sqrt(dx*dx + dy*dy)
                # If we grow r_i by dr and r_j by dr_j, they touch when d = r_i + dr + r_j + dr_j
                # Here we grow all circles somewhat uniformly or individually?
                # Simpler: Grow r_i such that it doesn't overlap with current r_j.
                # But other circles might grow too. 
                # Conservative approach: Assume neighbors don't grow in this step, 
                # or use a small step size.
                # Let's just ensure r_i + g < d - r_j
                limit = d - r[i] - r[j]
                if limit < g_max:
                    g_max = limit
            
            if g_max > 1e-8:
                # Grow by a fraction of the max available space to be safe
                growths[i] = min(g_max * 0.5, 0.05) # Cap growth per step
        
        # Apply growth
        r += growths
        
        # Re-check validity. If invalid, shrink back and stop.
        # To be fast, we can just check overlaps approximately or rely on the next force step.
        # But for the final result, we need strict validity.
        # Let's do a quick check.
        valid, val = _check_validity(x, y, r, w, h, tol=1e-6)
        if not valid:
            # Rollback
            r -= growths
            break

    return r

def _solve_for_aspect(w, h, rng_seed=0):
    """
    Solve for a specific aspect ratio w x h.
    """
    if w <= 0 or h <= 0:
        return None, 0.0
        
    rng = np.random.default_rng(rng_seed)
    
    # Initial guess: Grid
    side = int(math.ceil(math.sqrt(N)))
    # Try to fit N circles in a grid pattern
    # If N is not a perfect square, we might have uneven rows
    rows = side
    cols = side
    # Adjust to get exactly N
    # Simple heuristic: place in grid, if too many, remove corners? 
    # Better: Just place N circles in a grid of size ceil(sqrt(N))
    # And let the force solver fix the positions.
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    # Initialize positions
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N:
                break
            # Center of cell
            x[idx] = (j + 0.5) / cols * w
            y[idx] = (i + 0.5) / rows * h
            # Initial radius: small fraction of cell size
            cell_w = w / cols
            cell_h = h / rows
            r[idx] = min(cell_w, cell_h) * 0.4
            idx += 1
            
    # Add some randomness to break symmetry
    x += rng.uniform(-0.02, 0.02, N)
    y += rng.uniform(-0.02, 0.02, N)
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Force Directed Relaxation
    # Start with small dt, increase if stable? Or just fixed moderate dt.
    dt = 0.02
    k_overlap = 50.0
    k_wall = 20.0
    
    # Iterate
    max_iter = 2000
    last_valid_val = 0.0
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
            
        x, y = _force_step(x, y, r, w, h, dt=dt, k_overlap=k_overlap, k_wall=k_wall)
        
        # Periodically check validity and try to grow
        if it % 50 == 0:
            valid, val = _check_validity(x, y, r, w, h)
            if valid:
                last_valid_val = val
                # Try to grow radii slightly
                r = _grow_radii(x, y, r, w, h, max_iter=20, growth_factor=0.005)
                # Re-check after growth
                valid2, val2 = _check_validity(x, y, r, w, h)
                if valid2:
                    last_valid_val = val2
                else:
                    # Rollback growth if it broke validity
                    # We need to store old r. Simplest: just stop growing if it breaks.
                    pass
                    
    # Final polish
    valid, val = _check_validity(x, y, r, w, h)
    if valid:
         # Aggressive growth
         r = _grow_radii(x, y, r, w, h, max_iter=200)
         valid, val = _check_validity(x, y, r, w, h)
         if valid:
             return (x, y, r, w, h), val
             
    return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Explicit deadline guard
    
    best_val = 0.0
    best_result = None
    
    # Define a set of aspect ratios to try
    # Perimeter = 2(w+h) = 4 => w+h = 2
    # w ranges from 0.5 to 1.5 (symmetric)
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.55, 1.45
    ]
    
    rng_master = np.random.default_rng(42)
    
    for i, w in enumerate(aspect_ratios):
        if time.time() > deadline - 15:
            break
            
        h = 2.0 - w
        
        # Try multiple random seeds for each aspect ratio
        # If we have time, try 2 seeds per aspect ratio
        seeds_to_try = [i*100, i*100 + 1]
        
        for seed in seeds_to_try:
            if time.time() > deadline - 10:
                break
                
            res, val = _solve_for_aspect(w, h, rng_seed=seed)
            
            if res is not None and val > best_val:
                best_val = val
                best_result = res
                
                # If we find a very good solution, we might want to spend more time polishing it
                # But for now, we just keep the best.
                
    # Final check and formatting
    if best_result is None:
        # Fallback: simple grid
        # This should not happen, but just in case
        w = 1.0
        h = 1.0
        side = 5 # 5x5 grid = 25 slots, we take 21
        x = []
        y = []
        r = []
        for i in range(5):
            for j in range(5):
                if len(x) >= N:
                    break
                cx = (j + 0.5) / 5.0 * w
                cy = (i + 0.5) / 5.0 * h
                rad = 0.5/5.0 * 0.9
                x.append(cx)
                y.append(cy)
                r.append(rad)
        x = np.array(x)
        y = np.array(y)
        r = np.array(r)
        return np.column_stack((x, y, r))
    else:
        x, y, r, w, h = best_result
        # Shrink radii slightly to ensure strict disjoint interiors for the evaluator
        # The evaluator checks d >= r_i + r_j. 
        # Our check uses d < r_i + r_j - tol as invalid.
        # To be safe, shrink by 1e-5
        r = r * (1 - 1e-5)
        return np.column_stack((x, y, r))

# Entry point
if __name__ == "__main__":
    # Just for local testing
    res = construct()
    print(res.shape)
