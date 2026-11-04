"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_packing(x, y, r):
    """
    Evaluate the packing:
    1. Check walls.
    2. Check overlaps.
    3. Return (is_valid, sum_radii, min_margin).
    min_margin is the minimum of (dist_ij - r_i - r_j) and (wall_dist - r_i).
    """
    n = len(r)
    # Wall constraints
    min_wall_margin = np.min(np.minimum(np.minimum(x - r, 1 - x - r), np.minimum(y - r, 1 - y - r)))
    if min_wall_margin < -1e-8:
        return False, 0.0, min_wall_margin
    
    # Overlap constraints
    # Vectorized distance calculation
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    margins = dist - r_sum
    min_overlap_margin = np.min(margins)
    
    if min_overlap_margin < -1e-8:
        return False, 0.0, min_overlap_margin
        
    total_r = np.sum(r)
    return True, total_r, min_overlap_margin

def _grid_init(n, seed=0):
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.mgrid[0:side, 0:side]
    xs = xs.ravel()[:n]
    ys = ys.ravel()[:n]
    x = (xs + 0.5) / side
    y = (ys + 0.5) / side
    # Initial radius: slightly less than half spacing to be valid
    r = np.full(n, 0.5/side - 0.01)
    return x, y, r

def _random_init(n, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.2, 0.8, n)
    y = rng.uniform(0.2, 0.8, n)
    r = np.full(n, 0.04)
    return x, y, r

def _langevin_dynamics(x, y, r, deadline, seed=0):
    """
    Simulate hard disks with Langevin dynamics to pack them.
    Objective: Maximize sum(r).
    Method: 
    - Increase r slowly.
    - Move particles to resolve overlaps (gradient of overlap penalty).
    - Add noise to escape local minima.
    - Cool noise over time.
    """
    n = len(r)
    rng = np.random.RandomState(seed)
    
    # Initial state
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    # Ensure initial validity
    valid, total, _ = _eval_packing(x, y, r)
    if not valid:
        # Shrink to valid
        for i in range(n):
             max_r = min(x[i], 1-x[i], y[i], 1-y[i])
             for j in range(n):
                 if i==j: continue
                 d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                 max_r = min(max_r, (d - r[j]) * 0.99)
             r[i] = max(0.01, max_r * 0.9)
        # Re-center if pushed out? No, just clip positions
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
    best_total = np.sum(r)
    best_state = (x.copy(), y.copy(), r.copy())
    
    # Parameters
    # We want to grow r.
    # Start r at current level.
    # Target r: We don't know the max, so we just try to increase.
    # Strategy: 
    # 1. Fix r, move x,y to maximize min_margin (jamming).
    # 2. Increase r slightly, repeat.
    
    # However, Langevin on (x,y) with fixed r is standard jamming.
    # To maximize sum(r), we can treat r as a global variable that grows, 
    # or individual r's. 
    # Let's try: Global scale factor S. r_i = S * r0_i.
    # Or simpler: Individual r_i, but coupled.
    # Actually, for equal circles, global S is best. For unequal, it's harder.
    # Let's assume we can grow all r's proportionally or independently.
    # Independent growth is better for packing.
    
    # Algorithm:
    # 1. Initialize r to small values.
    # 2. Loop:
    #    a. Attempt to increase r slightly (e.g. +0.001).
    #    b. Run Langevin on (x,y) to resolve overlaps caused by larger r.
    #    c. If successful (valid), update best.
    #    d. If stuck (time), break.
    
    # Let's refine: 
    # We will run Langevin for a fixed number of steps or time slice.
    # The "force" is the gradient of the overlap penalty + repulsion from walls.
    # Potential V = sum_{i<j} max(0, r_i+r_j - dist_ij)^2 + sum_i wall_penalty_i^2
    # Force = -grad V.
    
    # To maximize sum r, we can just increase r and let the system find a valid config.
    # If it can't find one, we stop.
    
    # Initial r setup: Use a small base radius
    # Let's start from the input x,y,r which should be valid.
    
    # Adaptive growth
    growth_rate = 0.002
    min_growth = 0.0001
    current_growth = growth_rate
    
    # Langevin parameters
    dt = 0.01
    T = 0.1 # Temperature
    cooling = 0.995
    
    steps_per_growth = 500 # Number of Langevin steps per growth attempt
    
    while time.time() < deadline - 5:
        # 1. Grow radii
        r_new = r + current_growth
        # Cap at max possible (0.5)
        r_new = np.minimum(r_new, 0.45)
        
        # 2. Try to find valid x,y for r_new
        x_curr = x.copy()
        y_curr = y.copy()
        r_curr = r_new.copy()
        
        success = False
        # Run Langevin
        for step in range(steps_per_growth):
            if time.time() > deadline - 5:
                break
                
            # Calculate forces
            # Overlaps
            dx = x_curr[:, np.newaxis] - x_curr[np.newaxis, :]
            dy = y_curr[:, np.newaxis] - y_curr[np.newaxis, :]
            dist = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(dist, 1.0) # Avoid div by 0
            
            r_sum = r_curr[:, np.newaxis] + r_curr[np.newaxis, :]
            # Overlap amount: max(0, r_i + r_j - dist)
            # Force is repulsive: (r_i + r_j - dist) / dist * (dx, dy)
            # We use a soft potential: U = 0.5 * k * (overlap)^2
            # F = -k * overlap * grad(dist)
            
            # Vectorized force calculation
            # overlap_ij = max(0, r_sum - dist)
            overlaps = np.maximum(0, r_sum - dist)
            
            # Force from overlaps
            # F_x_ij = k * overlap_ij * (x_i - x_j) / dist_ij
            # Note: dist is symmetric.
            # We need to sum forces on i from all j.
            
            # To avoid division by zero, dist is >= 1e-6 effectively due to fill_diagonal
            # But if dist is 0, force is undefined. We filled with 1.0, so overlaps on diagonal are 0.
            
            # Force matrix F_xy[i,j] is force on i from j?
            # Let's compute force on i.
            # F_i = sum_j k * overlap_ij * (x_i - x_j) / dist_ij
            
            # Note: overlap_ij is 0 if not overlapping.
            # We can compute:
            # factor_ij = k * overlap_ij / dist_ij
            # F_x_i = sum_j factor_ij * (x_i - x_j)
            
            k = 10.0 # Spring constant
            
            # factor matrix
            # dist has 1.0 on diagonal, overlaps 0. So factor is 0 on diagonal.
            factor = k * overlaps / dist
            
            # F_x_i = sum_j factor[i,j] * (x_i - x_j)
            # = x_i * sum_j factor[i,j] - sum_j factor[i,j] * x_j
            # Let S_i = sum_j factor[i,j]
            # Let P_i = sum_j factor[i,j] * x_j
            
            S_x = np.sum(factor, axis=1)
            P_x = np.sum(factor * x_curr[np.newaxis, :], axis=1)
            F_x = x_curr * S_x - P_x
            
            S_y = np.sum(factor, axis=1)
            P_y = np.sum(factor * y_curr[np.newaxis, :], axis=1)
            F_y = y_curr * S_y - P_y
            
            # Wall forces
            # x_i < r_i -> push right
            # 1 - x_i < r_i -> push left
            # Soft wall: if x_i - r_i < 0, force = -k_wall * (x_i - r_i)
            # Let's use the same k for simplicity, or stronger.
            k_wall = 10.0
            
            # Left wall: x_i - r_i < 0
            viol_left = np.maximum(0, r_curr - x_curr)
            F_x -= k_wall * viol_left # Push right (positive x)
            
            # Right wall: 1 - x_i - r_i < 0
            viol_right = np.maximum(0, r_curr - (1 - x_curr))
            F_x += k_wall * viol_right # Push left (negative x)
            
            # Bottom wall: y_i - r_i < 0
            viol_bottom = np.maximum(0, r_curr - y_curr)
            F_y -= k_wall * viol_bottom
            
            # Top wall: 1 - y_i - r_i < 0
            viol_top = np.maximum(0, r_curr - (1 - y_curr))
            F_y += k_wall * viol_top
            
            # Langevin update
            # dx = F * dt + sqrt(2 * T * dt) * noise
            noise_x = rng.normal(0, 1, n)
            noise_y = rng.normal(0, 1, n)
            
            x_curr += F_x * dt + np.sqrt(2 * T * dt) * noise_x
            y_curr += F_y * dt + np.sqrt(2 * T * dt) * noise_y
            
            # Clip to hard bounds to prevent flying out (though forces should keep them in)
            # Actually, for hard disks, we can just let them move, but clipping helps stability
            # if forces are too weak.
            # But clipping can cause energy injection.
            # Let's just clip to [0,1] roughly.
            x_curr = np.clip(x_curr, 0.0, 1.0)
            y_curr = np.clip(y_curr, 0.0, 1.0)
            
            # Check validity (expensive, do less often)
            if step % 50 == 0:
                valid, total, margin = _eval_packing(x_curr, y_curr, r_curr)
                if valid:
                    # If valid, we can accept this state
                    success = True
                    # Optionally break early if margin is good? No, keep optimizing.
                    # But we want to accept the state and move to next growth.
                    # Actually, we should run the full steps_per_growth to settle.
                    # But if we are valid, we can store best.
                    if total > best_total:
                        best_total = total
                        best_state = (x_curr.copy(), y_curr.copy(), r_curr.copy())
                        
            # Cool temperature
            T *= cooling
            
        # End Langevin steps
        
        # After Langevin, check if we have a valid configuration
        valid, total, margin = _eval_packing(x_curr, y_curr, r_curr)
        
        if valid:
            x, y, r = x_curr, y_curr, r_curr
            if total > best_total:
                best_total = total
                best_state = (x.copy(), y.copy(), r.copy())
            # Reset temperature for next growth?
            # Or keep cooling?
            # If we keep cooling, it becomes deterministic.
            # Let's reset T to a moderate level to allow exploration for new r.
            T = 0.05 * (1 + current_growth) # Slightly higher T for larger growth?
            # Actually, as r grows, the system is tighter, so lower T is better.
            T = 0.02
            current_growth *= 0.95 # Slow down growth as we get denser
        else:
            # Failed to find valid config for this r
            # Try smaller growth? Or stop?
            # If we can't grow, we are at a local maximum for this topology.
            # Try to shake out the current configuration with higher T?
            # Or just accept current best and break?
            # Let's try a few more times with slightly smaller growth.
            current_growth *= 0.5
            if current_growth < min_growth:
                break
            
    return best_state[0], best_state[1], best_state[2], best_total

def _gradient_polish(x, y, r, deadline):
    """
    Final polish using scipy least_squares with vectorized residuals.
    """
    from scipy.optimize import least_squares
    
    n = len(r)
    
    def flatten(xy, r):
        return np.concatenate([xy.ravel(), r])
        
    def unflatten(z):
        xy = z[:2*n].reshape(n, 2)
        r = z[2*n:]
        return xy, r
        
    def residuals(z):
        xy, r = unflatten(z)
        r_safe = np.maximum(r, 1e-8)
        x, y = xy[:,0], xy[:,1]
        
        resids = []
        # Objective: -sum(r)
        resids.extend([-10.0 * r_safe])
        
        # Walls
        resids.append(np.maximum(0, r_safe - x))
        resids.append(np.maximum(0, r_safe - (1 - x)))
        resids.append(np.maximum(0, r_safe - y))
        resids.append(np.maximum(0, r_safe - (1 - y)))
        
        # Overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        r_sum = r_safe[:, np.newaxis] + r_safe[np.newaxis, :]
        violations = np.maximum(0, r_sum - dist)
        # Only add upper triangle to avoid duplicates? 
        # least_squares handles any shape.
        # But we want to minimize sum of squares.
        # Adding all pairs is fine, just doubles the weight.
        resids.append(violations)
        
        return np.concatenate(resids)
        
    z0 = flatten(x, y, r)
    
    try:
        res = least_squares(
            residuals, 
            z0, 
            method='lm', 
            max_nfev=2000,
            xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
    except Exception:
        return x, y, r, np.sum(r)
        
    xy_opt, r_opt = unflatten(res.x)
    r_opt = np.maximum(0, r_opt)
    # Clip positions
    for i in range(n):
        r_i = r_opt[i]
        xy_opt[i,0] = np.clip(xy_opt[i,0], r_i, 1-r_i)
        xy_opt[i,1] = np.clip(xy_opt[i,1], r_i, 1-r_i)
        
    valid, total, _ = _eval_packing(xy_opt[:,0], xy_opt[:,1], r_opt)
    if not valid:
        return x, y, r, np.sum(r)
        
    return xy_opt[:,0], xy_opt[:,1], r_opt, total

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_total = 0.0
    best_x = None
    best_y = None
    best_r = None
    
    # Strategy:
    # 1. Generate diverse initial configs (Grid, Random, Hex-like)
    # 2. Run Langevin dynamics on each to pack them.
    # 3. Polish the best results with gradient descent.
    
    # Initializers
    initializers = [
        ("grid", _grid_init),
        ("random", _random_init),
    ]
    
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    candidates = []
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        for name, init_func in initializers:
            if time.time() > deadline - 10:
                break
                
            try:
                x, y, r = init_func(N, seed=seed)
                # Run Langevin
                x_l, y_l, r_l, total_l = _langevin_dynamics(x, y, r, deadline, seed=seed)
                if total_l > best_total:
                    best_total = total_l
                    best_x = x_l.copy()
                    best_y = y_l.copy()
                    best_r = r_l.copy()
                candidates.append((total_l, x_l, y_l, r_l))
            except Exception:
                pass
                
    # Sort candidates by total
    candidates.sort(key=lambda c: c[0], reverse=True)
    
    # Polish top 3 candidates
    polished = []
    seen_keys = set()
    for c in candidates:
        if time.time() > deadline - 5:
            break
            
        total, x, y, r = c
        # Simple dedup
        key = (round(x[0],2), round(y[0],2), round(r[0],2))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        
        try:
            x_p, y_p, r_p, total_p = _gradient_polish(x, y, r, deadline)
            if total_p > best_total:
                best_total = total_p
                best_x = x_p.copy()
                best_y = y_p.copy()
                best_r = r_p.copy()
            polished.append((total_p, x_p, y_p, r_p))
        except Exception:
            pass
            
    # Final Fallback
    if best_x is None:
        x, y, r = _grid_init(N, 0)
        best_x, best_y, best_r = x, y, r
        best_total = np.sum(r)
        
    # Final Validation and Fix
    valid, total, _ = _eval_packing(best_x, best_y, best_r)
    if not valid:
        # Shrink radii
        for i in range(N):
            r_i = best_r[i]
            x_i = best_x[i]
            y_i = best_y[i]
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            for j in range(i+1, N):
                d = np.sqrt((best_x[i]-best_x[j])**2 + (best_y[i]-best_y[j])**2)
                max_r_overlap = (d - best_r[j]) * 0.99
                if r_i > max_r_overlap:
                    r_i = max_r_overlap
            best_r[i] = max(0.0, r_i * 0.99)
            
    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
# EVOLVE-BLOCK-END
