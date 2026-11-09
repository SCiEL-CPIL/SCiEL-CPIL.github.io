"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
import math

N = 26
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds before hard limit

def _compute_score(circles):
    """Compute sum of radii for a valid packing, or -inf if invalid."""
    n = len(circles)
    if n == 0:
        return 0.0

    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Check all radii are positive
    if np.any(r <= 0):
        return -np.inf

    # Check all circles are inside the unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf

    # Check no overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(x[i] - x[j], y[i] - y[j])
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf

    return np.sum(r)

def _max_radius(x, y, xs, ys, rs, i):
    """
    Calculate the maximum radius for circle i at position (x, y)
    given the positions and radii of all other circles.
    """
    # Max radius limited by walls
    r = min(x, 1.0 - x, y, 1.0 - y)
    
    if r <= 1e-9:
        return 1e-9

    # Max radius limited by other circles
    # Vectorized distance calculation
    dx = xs - x
    dy = ys - y
    dists = np.sqrt(dx*dx + dy*dy)
    
    # dists[i] is 0, we need to exclude i
    # r_i must satisfy dists[j] >= r_i + r_j  => r_i <= dists[j] - r_j
    # So r_i <= min(dists[j] - r_j) for all j != i
    
    # Create a mask to exclude i
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    
    # Calculate constraint radii
    constrained_r = dists[mask] - rs[mask]
    
    if np.any(constrained_r):
        r = min(r, np.min(constrained_r))
        
    return max(r, 1e-9)

def _local_search(xs, ys, rs, deadline):
    """
    Local search: iterate through circles, try to move each one to a position
    that allows for a larger radius.
    """
    best_score = np.sum(rs)
    n = N
    rng = np.random.RandomState(42)
    
    # We will perform multiple passes
    # In each pass, we try to improve each circle
    # If no improvement in a full pass, we do some random jumps
    
    while time.time() < deadline:
        improved = False
        
        for i in range(n):
            if time.time() >= deadline:
                return xs, ys, rs, best_score
            
            old_r = rs[i]
            old_x, old_y = xs[i], ys[i]
            
            # Try small perturbations around current position
            # and some random jumps
            best_r = old_r
            best_x, best_y = old_x, old_y
            
            # 1. Small local search (grid-like or random steps)
            # We try a few random steps in a small neighborhood
            for _ in range(30):
                # Random step size, decreasing? No, fixed small step for local search
                step = 0.02
                dx = rng.uniform(-step, step)
                dy = rng.uniform(-step, step)
                
                nx = old_x + dx
                ny = old_y + dy
                
                # Clamp to valid range for the *current* radius to ensure feasibility
                # Actually, we want to find a new radius. 
                # If we move to (nx, ny), the max radius is _max_radius(nx, ny, ...)
                # But we must ensure the circle doesn't overlap *during* the move? 
                # No, the move is instantaneous in the model. 
                # However, if we pick a position that is currently overlapping, 
                # the _max_radius function will return a small radius (limited by overlap).
                # We want to find a position where the *potential* radius is high.
                
                # Clamp position to be inside the square (center can be anywhere, 
                # radius will be limited by walls)
                if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                    continue
                    
                new_r = _max_radius(nx, ny, xs, ys, rs, i)
                
                # We only accept if new_r is significantly larger than old_r
                # Note: moving to a position with a smaller radius is bad for sum, 
                # but might allow others to grow? 
                # This simple local search maximizes *individual* radius.
                # This is a greedy approach. 
                if new_r > best_r + 1e-8:
                    best_r = new_r
                    best_x, best_y = nx, ny
            
            # 2. Random jump to escape local minima
            # Occasionally try a larger jump
            if rng.rand() < 0.1:
                for _ in range(10):
                    nx = rng.uniform(0.05, 0.95)
                    ny = rng.uniform(0.05, 0.95)
                    new_r = _max_radius(nx, ny, xs, ys, rs, i)
                    if new_r > best_r + 1e-8:
                        best_r = new_r
                        best_x, best_y = nx, ny
            
            if best_r > old_r + 1e-8:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                current_score = np.sum(rs)
                if current_score > best_score:
                    best_score = current_score
        
        # If no improvement in a full pass, we might be stuck.
        # The random jumps inside the loop help, but if we stagnate completely, 
        # we can break or just continue until time runs out.
        # For now, we just continue until deadline.
        
    return xs, ys, rs, best_score

def _make_initial_grid(n):
    """Create initial grid layout with strict non-overlap."""
    side = int(np.ceil(np.sqrt(n)))
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Shrink radius slightly to ensure strict disjoint interiors
            # In a grid, max radius is 0.5/side. 
            # To be safe, use 0.45/side
            r = 0.45 / side
            xs.append(cx)
            ys.append(cy)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)

def _make_initial_random(n, seed=42):
    """Create random initial layout with small radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    r = np.full(n, 0.02)
    return x, y, r

def construct():
    """Construct 26 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    best_circles = None
    best_score = -np.inf

    # Try multiple initial configurations
    # 1. Grid
    xs, ys, rs = _make_initial_grid(N)
    # Validate initial grid
    init_circles = np.column_stack([xs, ys, rs])
    init_score = _compute_score(init_circles)
    if not np.isinf(init_score) and init_score > best_score:
        best_score = init_score
        best_circles = init_circles.copy()
        
    # Run local search on grid
    remaining = deadline - time.time()
    if remaining > 2.0:
        xs_opt, ys_opt, rs_opt, score_opt = _local_search(xs, ys, rs, deadline)
        circles_opt = np.column_stack([xs_opt, ys_opt, rs_opt])
        # Final validation
        if _compute_score(circles_opt) > best_score:
            best_score = score_opt
            best_circles = circles_opt.copy()

    # 2. Random restarts
    # Use remaining time for random restarts
    for seed in range(10):
        if time.time() > deadline - 2.0:
            break
            
        xs_r, ys_r, rs_r = _make_initial_random(N, seed=seed+100)
        
        # Quick validation of random init (should be valid if r is small enough)
        # r=0.02, min dist in random is not guaranteed, but likely valid for small r
        # Let's just run local search which will fix overlaps by reducing radius if needed
        # Actually, _max_radius ensures validity. 
        # But if initial config is invalid (overlapping), _max_radius for one circle 
        # might be limited by the other, leading to a valid but potentially suboptimal start.
        # To be safe, let's check if initial is valid. If not, shrink radii.
        init_c = np.column_stack([xs_r, ys_r, rs_r])
        if _compute_score(init_c) == -np.inf:
             # Shrink radii until valid
             scale = 0.5
             while scale > 0.01:
                 rs_r = rs_r * scale
                 init_c = np.column_stack([xs_r, ys_r, rs_r])
                 if _compute_score(init_c) != -np.inf:
                     break
                 scale *= 0.5
        
        # Run local search
        # Allocate time based on remaining
        # Simple: run until deadline or no improvement? 
        # _local_search runs until deadline. 
        # This is bad if we have multiple restarts. 
        # We need to limit time per restart.
        
        time_per_restart = max(1.0, (deadline - time.time()) / 5.0) # Approx
        restart_deadline = time.time() + time_per_restart
        
        xs_opt, ys_opt, rs_opt, score_opt = _local_search(xs_r, ys_r, rs_r, restart_deadline)
        circles_opt = np.column_stack([xs_opt, ys_opt, rs_opt])
        
        # Validate
        final_score = _compute_score(circles_opt)
        if final_score != -np.inf and final_score > best_score:
            best_score = final_score
            best_circles = circles_opt.copy()

    # If no valid result, fall back to grid
    if best_circles is None:
        xs, ys, rs = _make_initial_grid(N)
        best_circles = np.column_stack([xs, ys, rs])

    # Final validation
    final_score = _compute_score(best_circles)
    if np.isinf(final_score):
        # As a last resort, return a very safe small circle configuration
        xs, ys, rs = _make_initial_grid(N)
        best_circles = np.column_stack([xs, ys, rs])

    return best_circles
