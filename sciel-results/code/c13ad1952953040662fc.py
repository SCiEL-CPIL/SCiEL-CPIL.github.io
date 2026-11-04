"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def _feasible(circles, tol=1e-7):
    """Check if circles are valid: disjoint interiors, inside a perimeter-4 rectangle."""
    if len(circles) != N:
        return False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    # Check radii positive
    if np.any(r <= 0):
        return False
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    # Check bounding box
    x_min = cx - r
    x_max = cx + r
    y_min = cy - r
    y_max = cy + r
    
    W = x_max.max() - x_min.min()
    H = y_max.max() - y_min.min()
    
    if 2 * (W + H) > 4.0 + tol:
        return False
        
    return True

def _evaluate(circles):
    """Compute the objective value: sum of radii, 0 if infeasible."""
    if not _feasible(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _scale_to_perimeter(circles):
    """Scale circles uniformly so that the bounding rectangle has perimeter exactly 4 (or less if already)."""
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    x_min = (cx - r).min()
    x_max = (cx + r).max()
    y_min = (cy - r).min()
    y_max = (cy + r).max()
    
    W = x_max - x_min
    H = y_max - y_min
    
    if W + H == 0:
        return circles.copy()
        
    # We want 2*(W' + H') = 4 => W' + H' = 2
    # Current W + H = S
    # Scale factor s = 2 / S
    # New positions: x' = x_min + s*(x - x_min), r' = s*r
    # This preserves the shape but scales it.
    
    s = 2.0 / (W + H)
    
    # If s > 1, we are expanding. We must ensure circles don't overlap after expansion.
    # Since expansion increases distances between centers and radii by same factor,
    # relative geometry is preserved. If they were disjoint before, they are disjoint after.
    # Wait, if they were *touching*, they are still touching. If they were *overlapping*, they are still overlapping.
    # So scaling preserves feasibility of non-overlap.
    # However, we must ensure the scaled configuration fits in the new box.
    # By construction, the new box has width W*s and height H*s, and the circles fit exactly in it.
    
    new_cx = x_min + s * (cx - x_min)
    new_cy = y_min + s * (cy - y_min)
    new_r = s * r
    
    return np.column_stack((new_cx, new_cy, new_r))

def _random_grid_init():
    """Generate a random grid-like initialization."""
    # Use a 5x5 grid but jittered
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            # Random offset within cell
            dx = np.random.uniform(0.1, 0.9) / side
            dy = np.random.uniform(0.1, 0.9) / side
            x = i / side + dx
            y = j / side + dy
            positions.append([x, y])
    
    positions = np.array(positions[:N])
    
    # Initial radii: estimate based on grid spacing
    # Roughly 1/side, but we will optimize
    r0 = 0.18
    radii = np.full(N, r0)
    
    circles = np.column_stack((positions[:, 0], positions[:, 1], radii))
    
    # Scale to perimeter 4
    circles = _scale_to_perimeter(circles)
    
    return circles

def _local_optimize(circles, max_iters=2000, dt=0.05):
    """
    Gradient-free local optimization using a simple repulsion/attraction scheme.
    We maximize sum(r) subject to non-overlap and perimeter constraint.
    
    Strategy:
    1. Fix the rectangle size to perimeter 4 by scaling.
    2. Use a penalty method or direct gradient descent on positions and radii.
    
    Simpler approach:
    - Start with a feasible configuration (scaled to perim 4).
    - Iteratively try to increase radii and adjust positions to maintain feasibility.
    
    Let's use a simple simulated annealing or random hill climbing with scaling.
    """
    best_circles = circles.copy()
    best_val = _evaluate(best_circles)
    
    # If initial is infeasible, try to fix it by shrinking
    if best_val == 0.0:
        # Shrink radii until feasible
        for _ in range(100):
            circles[:, 2] *= 0.9
            if _feasible(circles):
                break
        best_circles = circles.copy()
        best_val = _evaluate(best_circles)
        if best_val == 0.0:
            return best_circles, best_val

    for i in range(max_iters):
        # Pick a random circle
        idx = np.random.randint(N)
        
        # Try to increase its radius slightly
        dr = np.random.uniform(0.001, 0.01)
        new_r = best_circles[idx, 2] + dr
        
        if new_r <= 0:
            continue
            
        # Try to move it slightly to accommodate the larger radius
        # Or just check if we can just increase radius
        test_circles = best_circles.copy()
        test_circles[idx, 2] = new_r
        
        # Check feasibility
        if _feasible(test_circles):
            best_circles = test_circles
            best_val += dr
        else:
            # Try moving the circle
            # Move in a random direction
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.01, 0.05)
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            test_circles[idx, 0] += dx
            test_circles[idx, 1] += dy
            test_circles[idx, 2] = new_r
            
            if _feasible(test_circles):
                best_circles = test_circles
                best_val += dr
            else:
                # Try just moving without increasing radius (to help other circles)
                test_circles = best_circles.copy()
                test_circles[idx, 0] += dx
                test_circles[idx, 1] += dy
                
                if _feasible(test_circles):
                    best_circles = test_circles
                    # Value doesn't change, but position might allow future growth
    
    return best_circles, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_circles = None
    best_val = 0.0
    
    # 1. Seed initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    seed_circles = np.array(cells[:N])
    
    # Scale seed to perimeter 4
    seed_circles = _scale_to_perimeter(seed_circles)
    seed_val = _evaluate(seed_circles)
    
    if seed_val > best_val:
        best_val = seed_val
        best_circles = seed_circles.copy()
    
    # 2. Local optimization on seed
    opt_circles, opt_val = _local_optimize(seed_circles, max_iters=5000)
    if opt_val > best_val:
        best_val = opt_val
        best_circles = opt_circles.copy()
    
    # 3. Random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        init_circles = _random_grid_init()
        
        # Check if time is running out, use fewer iterations
        remaining = deadline - time.time()
        if remaining < 10:
            iters = 1000
        elif remaining < 30:
            iters = 3000
        else:
            iters = 5000
            
        opt_circles, opt_val = _local_optimize(init_circles, max_iters=iters)
        
        if opt_val > best_val:
            best_val = opt_val
            best_circles = opt_circles.copy()
            
        # If we found a very good solution, we might break early? 
        # Theoretical max is around 21 * (2/sqrt(21)) / 2? No.
        # If we have 21 circles in a 1x1 square (perim 4), max radius is 0.5/sqrt(21) ~ 0.109. Sum ~ 2.28.
        # But we can shape the rectangle.
        # Let's just keep searching until time runs out.
        
    return best_circles

# EVOLVE-BLOCK-END
