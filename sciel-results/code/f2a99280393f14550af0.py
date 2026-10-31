"""Circle packing in perimeter-4 rectangle: Force-Driven Radius Maximization."""
import numpy as np
import time
import random

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _force_optimize(x0, y0, r0, W, H, time_budget, stiffness=500.0):
    """
    Optimize positions and radii using force-directed relaxation.
    We maximize radii by applying outward forces, but we must keep them inside
    the rectangle and non-overlapping.
    
    Strategy:
    1. Start with small radii.
    2. Apply "growth" forces to increase radii.
    3. Apply "repulsion" forces to resolve overlaps (reducing effective growth or moving centers).
    4. Apply "wall" forces to keep centers inside.
    
    Actually, a better approach for maximizing sum of radii with fixed W,H:
    Treat r_i as variables. The objective is sum(r_i).
    Constraints:
    1. x_i in [r_i, W-r_i], y_i in [r_i, H-r_i]
    2. dist(i,j) >= r_i + r_j
    
    We can use a gradient ascent approach on the "feasible" region.
    However, force-directed methods usually minimize energy.
    Let's minimize: E = sum(r_i)^2 * penalty(overlaps) - sum(r_i)
    Or simpler: 
    Let's use a "push" method.
    For each circle, we want to expand it.
    If expanding causes overlap, we push the centers apart.
    
    Algorithm:
    1. Initialize x, y, r.
    2. Loop:
       a. For each circle i, calculate max possible r_i given current x,y and other r_j.
          r_i_max = min(
              x_i, W-x_i, y_i, H-y_i,
              min_j( dist(i,j) - r_j )
          )
       b. If r_i < r_i_max, set r_i = r_i_max.
       c. This might cause new overlaps or wall violations if we just set r.
       
    This is the "polish" step. To get a good starting point for polish, we need good positions.
    
    Let's use a hybrid:
    1. Force-directed relaxation to find a valid packing with fixed radii (e.g., small uniform radii).
    2. Then iteratively increase radii.
    
    Actually, the "polish" step is very effective if the positions are close to optimal.
    The bottleneck is finding positions where the "bottleneck" distances are maximized.
    
    Let's try a different force model:
    Minimize E = sum( max(0, (r_i + r_j - dist_ij)^2 ) ) + sum( max(0, (r_i - x_i)^2) + ... )
    But we want to MAXIMIZE sum(r_i).
    
    Let's use a "simulated annealing" style or just a simple gradient ascent on r with position adjustment.
    
    Revised Strategy:
    1. Initialize positions randomly or hexagonally.
    2. Set initial radii small.
    3. Iteration:
       - For each circle, compute the "limiting" distance to walls and neighbors.
       - Update radius to this limit (clamped to current radius + step).
       - If this causes overlaps (due to simultaneous updates), use a force-directed step to separate centers.
    
    Let's implement `_expand_and_resolve`:
    1. For each i, compute r_i_potential = min(dist_to_walls, min_j(dist_ij - r_j)).
    2. Set r_i = r_i_potential.
    3. This creates overlaps.
    4. Run force-directed relaxation on positions ONLY (radii fixed) to resolve overlaps.
    5. Repeat.
    """
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    
    start = time.time()
    
    for _ in range(2000):
        if time.time() - start > time_budget:
            break
        
        # Step 1: Expand radii to maximum possible given current positions
        for i in range(N):
            r_max = min(x[i], W - x[i], y[i], H - y[i])
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_max = min(r_max, d - r[j])
            r[i] = max(r[i], r_max) # Monotonic increase? No, positions change.
            # Actually, if we move positions, r_max might decrease.
            # But we are fixing r for the next force step.
        
        # Step 2: Force-directed relaxation to resolve overlaps created by expansion
        # We want to move centers to satisfy dist(i,j) >= r_i + r_j
        # and stay within walls.
        
        # Use a simple iterative solver
        for _iter in range(50):
            if time.time() - start > time_budget:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall constraints (soft)
            for i in range(N):
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10
                if x[i] > W - r[i]:
                    fx[i] -= (x[i] - (W - r[i])) * 10
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10
                if y[i] > H - r[i]:
                    fy[i] -= (y[i] - (H - r[i])) * 10
            
            # Overlap constraints (hard push)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d_sq = dx*dx + dy*dy
                    d = np.sqrt(d_sq) if d_sq > 1e-10 else 1e-10
                    
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        # Force proportional to overlap
                        force = overlap * 50
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
            
            # Update positions
            # Adaptive step size
            max_force = np.max(np.sqrt(fx*fx + fy*fy))
            if max_force < 1e-6:
                break
            step = 0.1 / max_force * 0.5 # Damped step
            x += fx * step
            y += fy * step
            
            # Clamp to hard boundaries (with radius margin)
            x = np.clip(x, r, W - r)
            y = np.clip(y, r, H - r)
            
            # If clamping caused new overlaps, the next iteration will fix it.
            
        # Check if we can expand further? 
        # The loop above already sets r to max. 
        # But positions moved, so r_max might be different.
        # The outer loop handles re-expansion.
        
    return x, y, r

def _polish_radii(x, y, r, w, h, time_budget, max_rounds=100):
    """
    Iteratively expand radii to their exact geometric maximum.
    This is a local search that converges to a local optimum where all circles
    are "tight" against walls or neighbors.
    """
    start = time.time()
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            # Max radius limited by walls
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            if r_max > r[i] + 1e-12:
                r[i] = max(r_max, 1e-12)
                improved = True
        
        if not improved:
            break
    
    return x, y, r

def _hexagonal_init(w, h, n, seed=0):
    """Generate a hexagonal close-packed layout."""
    random.seed(seed)
    
    # Estimate optimal radius
    r_est = np.sqrt((w * h) / (n * 2 * np.sqrt(3)))
    
    # Determine grid dimensions
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal offset
        offset = 0.5 * (row % 2)
        x[i] = (col + 0.5 + offset) * (w / (cols + 0.5))
        y[i] = (row + 0.5) * (h / rows)
    
    # Calculate initial radius (small to allow expansion)
    min_dx = w / (cols + 0.5)
    min_dy = h / rows
    r_init = min(min_dx, min_dy) * 0.2
    r_init = min(r_init, w/2, h/2)
    
    r = np.zeros(n)
    for i in range(n):
        r[i] = min(r_init, x[i], w - x[i], y[i], h - y[i])
    
    # Add small random perturbation to break symmetry
    if seed > 0:
        noise = 0.01 * np.random.random(n)
        x = np.clip(x + (noise - 0.5) * 0.02, r, w - r)
        y = np.clip(y + (noise - 0.5) * 0.02, r, h - r)
    
    return x, y, r

def _random_init(w, h, n, seed=0):
    """Generate a random initial packing."""
    random.seed(seed)
    np.random.seed(seed)
    
    x = np.random.uniform(0, w, n)
    y = np.random.uniform(0, h, n)
    r = np.full(n, 0.01)
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Search over aspect ratios
    # Perimeter 4 => 2(w+h) = 4 => w+h = 2
    # Square is 1x1.
    # Try a range of aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 2.0, 0.4]
    
    for w in aspect_ratios:
        if time.time() > deadline - 30:
            break
        h = 2.0 - w
        if h < 0.3: # Too narrow
            continue
            
        # Try multiple initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            if seed == 0:
                x, y, r = _hexagonal_init(w, h, N, seed=0)
            elif seed == 1:
                x, y, r = _hexagonal_init(w, h, N, seed=1)
            elif seed == 2:
                x, y, r = _random_init(w, h, N, seed=seed*100)
            elif seed == 3:
                x, y, r = _random_init(w, h, N, seed=seed*100+1)
            else:
                x, y, r = _hexagonal_init(w, h, N, seed=seed)
            
            # Force-driven optimization
            remaining_time = deadline - time.time() - 5
            if remaining_time > 2:
                x, y, r = _force_optimize(x, y, r, w, h, time_budget=remaining_time)
                
                # Polish radii
                x, y, r = _polish_radii(x, y, r, w, h, time_budget=5)
                update_best(x, y, r, w, h)
                
                # Second pass of force optimization starting from polished result
                # This can help escape local minima where polish got stuck
                remaining_time = deadline - time.time() - 5
                if remaining_time > 2:
                    x, y, r = _force_optimize(x, y, r, w, h, time_budget=remaining_time)
                    x, y, r = _polish_radii(x, y, r, w, h, time_budget=5)
                    update_best(x, y, r, w, h)
    
    # Fallback
    if best_result is None:
        w, h = 1.0, 1.0
        x, y, r = _hexagonal_init(w, h, N, seed=0)
        x, y, r = _force_optimize(x, y, r, w, h, time_budget=5)
        x, y, r = _polish_radii(x, y, r, w, h, time_budget=2)
        best_result = (x, y, r, w, h)
    
    x, y, r, w, h = best_result
    
    # Final safety shrink to ensure validity
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
