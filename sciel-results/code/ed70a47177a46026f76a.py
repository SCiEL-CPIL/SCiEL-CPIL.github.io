"""Circle packing in perimeter-4 rectangle: Physics-based Relaxation."""
import numpy as np
import time
import random

N = 21

def _check_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Vectorized overlap check
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    # Check upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol)**2):
        return False
    return True

def _relax_physics(cx, cy, r, W, H, step, iterations=1000, time_limit=None, start_time=None):
    """
    Physics-based relaxation.
    W + H = 2 is fixed. We vary W.
    """
    if start_time is None:
        start_time = time.time()
    
    best_val = -np.inf
    best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
    
    # Initial setup
    # We will simulate forces.
    # To maximize sum(r), we can think of this as:
    # 1. Keep radii fixed, move centers to resolve overlaps.
    # 2. If no overlaps, grow radii slightly.
    # 3. Adjust W to balance pressure.
    
    # Alternatively, a more direct physics model:
    # Each circle has a "target" radius. We start small and grow.
    # But simpler: 
    # 1. Resolve overlaps by moving centers.
    # 2. If valid, try to increase all radii by a small factor.
    # 3. If that causes overlap, resolve again.
    # 4. Adjust W based on which walls are more "active".
    
    current_r = r.copy()
    current_cx = cx.copy()
    current_cy = cy.copy()
    current_W = W
    current_H = H
    
    # Adaptive parameters
    lr_move = 0.05
    lr_grow = 0.001
    lr_aspect = 0.01
    
    for it in range(iterations):
        if time.time() - start_time > (time_limit if time_limit else 110.0):
            break
            
        # 1. Resolve overlaps
        # Iterate a few times per outer step to settle
        for _ in range(5):
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            overlap_found = False
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = current_cx[i] - current_cx[j]
                    dy = current_cy[i] - current_cy[j]
                    d2 = dx*dx + dy*dy
                    d = np.sqrt(d2) if d2 > 1e-12 else 1e-12
                    rs = current_r[i] + current_r[j]
                    
                    if d < rs:
                        overlap_found = True
                        overlap = rs - d
                        # Force proportional to overlap
                        f = overlap * 0.5
                        if d > 1e-12:
                            fx = f * dx / d
                            fy = f * dy / d
                            forces_x[i] += fx
                            forces_y[i] += fy
                            forces_x[j] -= fx
                            forces_y[j] -= fy
                        else:
                            # If coincident, push randomly
                            angle = random.uniform(0, 2*np.pi)
                            forces_x[i] += f * np.cos(angle)
                            forces_y[i] += f * np.sin(angle)
                            forces_x[j] -= f * np.cos(angle)
                            forces_y[j] -= f * np.sin(angle)
            
            if not overlap_found:
                break
                
            # Apply forces
            current_cx += forces_x * lr_move
            current_cy += forces_y * lr_move
            
            # Project back to bounds
            # Hard projection to ensure they stay inside
            current_cx = np.clip(current_cx, current_r, current_W - current_r)
            current_cy = np.clip(current_cy, current_r, current_H - current_r)
            
        # 2. Check if we can grow
        # Try to grow all radii by a small amount
        growth_factor = 1.0 + lr_grow
        new_r = current_r * growth_factor
        
        # Check if new radii are valid with current positions
        # We need to check if they fit in bounds and don't overlap
        valid_growth = True
        # Bounds check for growth
        if np.any(new_r > current_cx) or np.any(new_r > current_W - current_cx):
             valid_growth = False
        if np.any(new_r > current_cy) or np.any(new_r > current_H - current_cy):
             valid_growth = False
             
        if valid_growth:
             # Check overlaps with new radii
             # Re-use vectorized check
             dx = current_cx[:, np.newaxis] - current_cx[np.newaxis, :]
             dy = current_cy[:, np.newaxis] - current_cy[np.newaxis, :]
             d2 = dx**2 + dy**2
             rs = new_r[:, np.newaxis] + new_r[np.newaxis, :]
             mask = np.triu(np.ones((N, N), dtype=bool), k=1)
             if np.any(d2[mask] < (rs[mask] - 1e-9)**2):
                 valid_growth = False
                 
        if valid_growth:
            current_r = new_r
            val = np.sum(current_r)
            if val > best_val:
                best_val = val
                best_state = (current_cx.copy(), current_cy.copy(), current_r.copy(), current_W, current_H)
            
            # Decay growth rate slightly to avoid instability
            lr_grow *= 0.999
        else:
            # If can't grow, reduce growth rate
            lr_grow *= 0.9
            # Also, maybe we should try to move circles to create space?
            # The force resolution above handles this partially.
            # We can also try to "jiggle" the system to escape local minima
            if it % 50 == 0:
                jiggle = 0.01 * (1.0 - it/iterations)
                current_cx += np.random.normal(0, jiggle, N)
                current_cy += np.random.normal(0, jiggle, N)
                current_cx = np.clip(current_cx, current_r, current_W - current_r)
                current_cy = np.clip(current_cy, current_r, current_H - current_r)
                
        # 3. Adjust Aspect Ratio
        # Measure "pressure" on walls
        # Pressure is roughly proportional to how close circles are to walls
        # Or simply: if we are hitting left/right more than top/bottom, widen W.
        
        # Calculate distance to walls
        dist_left = current_cx - current_r
        dist_right = current_W - current_cx - current_r
        dist_bottom = current_cy - current_r
        dist_top = current_H - current_cy - current_r
        
        # Sum of violations (should be 0 if projected, but we want to know "tendency")
        # Actually, since we project, dists are >= 0.
        # We want to minimize the "tightness".
        # If circles are squeezed in X, dist_left and dist_right are small.
        # If squeezed in Y, dist_bottom and dist_top are small.
        
        # A simple heuristic: 
        # If average distance to X walls < average distance to Y walls, we are tighter in X.
        # So we should increase W (and decrease H) to give more room in X.
        avg_dist_x = (np.mean(dist_left) + np.mean(dist_right)) / 2.0
        avg_dist_y = (np.mean(dist_bottom) + np.mean(dist_top)) / 2.0
        
        if avg_dist_x < avg_dist_y:
            # Tighter in X, widen X
            current_W += lr_aspect
        else:
            # Tighter in Y, widen Y (decrease W)
            current_W -= lr_aspect
            
        # Clamp W to valid range (e.g., 0.5 to 1.5)
        current_W = np.clip(current_W, 0.5, 1.5)
        current_H = 2.0 - current_W
        
        # Re-project positions to new bounds
        current_cx = np.clip(current_cx, current_r, current_W - current_r)
        current_cy = np.clip(current_cy, current_r, current_H - current_r)

    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_result = None
    
    # Try multiple initial configurations
    # 1. Grid
    # 2. Random
    # 3. Specific aspect ratios
    
    configs = []
    
    # Grid layouts
    grid_dims = [(3, 7), (7, 3), (4, 6), (6, 4), (5, 5)]
    for rows, cols in grid_dims:
        if rows * cols < N:
            continue
        W_init = 1.0 # Start square
        H_init = 1.0
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        cell_w = W_init / cols
        cell_h = H_init / rows
        for i in range(N):
            row = i // cols
            col = i % cols
            cx[i] = (col + 0.5) * cell_w
            cy[i] = (row + 0.5) * cell_h
            r[i] = min(cell_w, cell_h) * 0.4
        configs.append((cx, cy, r, W_init, H_init))
        
    # Random initializations with different aspect ratios
    for W_init in [0.8, 1.0, 1.2]:
        H_init = 2.0 - W_init
        cx = np.random.uniform(0.1, W_init - 0.1, N)
        cy = np.random.uniform(0.1, H_init - 0.1, N)
        # Start with small radii to avoid massive initial overlap
        r = np.full(N, 0.1)
        configs.append((cx, cy, r, W_init, H_init))
        
    for cx, cy, r, W, H in configs:
        if time.time() > deadline - 10:
            break
            
        # Run physics relaxation
        res_cx, res_cy, res_r, res_W, res_H = _relax_physics(
            cx, cy, r, W, H, 
            step=1.0, 
            iterations=2000, 
            time_limit=10.0, 
            start_time=time.time()
        )
        
        if _check_valid(res_cx, res_cy, res_r, res_W, res_H):
            val = np.sum(res_r)
            if val > best_val:
                best_val = val
                best_result = (res_cx, res_cy, res_r, res_W, res_H)
                
    if best_result is None:
        # Fallback: simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r, W, H = best_result
    
    # Final Polish: Local Search
    # Try to squeeze out more by growing radii and adjusting positions
    for _ in range(500):
        if time.time() > deadline - 2:
            break
        # Try growing radii
        r_trial = r * 1.0005
        if _check_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            # Try moving a random circle
            idx = np.random.randint(N)
            cx_trial = cx.copy()
            cy_trial = cy.copy()
            cx_trial[idx] += np.random.normal(0, 0.005)
            cy_trial[idx] += np.random.normal(0, 0.005)
            cx_trial[idx] = np.clip(cx_trial[idx], r[idx], W - r[idx])
            cy_trial[idx] = np.clip(cy_trial[idx], r[idx], H - r[idx])
            
            if _check_valid(cx_trial, cy_trial, r, W, H):
                cx = cx_trial
                cy = cy_trial
                # Try growing again after move
                r_trial = r * 1.0005
                if _check_valid(cx, cy, r_trial, W, H):
                    r = r_trial
            # Also try adjusting aspect ratio slightly
            W_trial = W + np.random.normal(0, 0.01)
            W_trial = np.clip(W_trial, 0.5, 1.5)
            H_trial = 2.0 - W_trial
            if _check_valid(cx, cy, r, W_trial, H_trial):
                # Check if it allows growth
                r_trial2 = r * 1.0005
                if _check_valid(cx, cy, r_trial2, W_trial, H_trial):
                    W = W_trial
                    H = H_trial
                    r = r_trial2

    # Final safety check
    if not _check_valid(cx, cy, r, W, H):
        # Shrink until valid
        scale = 1.0
        while not _check_valid(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _check_valid(cx, cy, r, W, H):
             # Extreme fallback
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
