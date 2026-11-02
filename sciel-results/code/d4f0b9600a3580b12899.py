"""Circle packing in perimeter-4 rectangle: Simulated Annealing with Penalty."""
import numpy as np
import time
import random

N = 21

def _energy(cx, cy, r, W, H):
    """
    Calculate the penalty energy of the configuration.
    Lower energy is better. Energy is 0 if valid.
    """
    # 1. Overlap penalty
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d = np.sqrt(dx**2 + dy**2 + 1e-12)
    # Required distance
    req = r[:, np.newaxis] + r[np.newaxis, :]
    # Overlap amount (positive if overlapping)
    overlap = np.maximum(0, req - d)
    # Exclude self-interaction
    np.fill_diagonal(overlap, 0)
    overlap_penalty = np.sum(overlap**2)
    
    # 2. Boundary penalty
    # Distance to walls
    dx_left = cx - r
    dx_right = W - cx - r
    dy_bottom = cy - r
    dy_top = H - cy - r
    
    # Violation (positive if outside)
    v_left = np.maximum(0, -dx_left)
    v_right = np.maximum(0, -dx_right)
    v_bottom = np.maximum(0, -dy_bottom)
    v_top = np.maximum(0, -dy_top)
    
    boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
    
    # Weight boundary heavily to ensure validity
    return overlap_penalty + 10.0 * boundary_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_state = None
    
    # Initial configuration: Grid-like with some randomness
    # We want to start with a reasonable aspect ratio.
    # Perimeter 4 => W+H=2. Square is 1x1.
    
    # We will run multiple SA chains or one long one with restarts.
    # Let's do a single long SA chain with adaptive moves.
    
    # Initial state
    # Grid 5x5 = 25 slots, we take 21.
    # W=1, H=1.
    W = 1.0
    H = 1.0
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    # Place in a 5x5 grid, skipping 4 corners or similar to maximize space
    # Actually, just random init is safer for SA to escape bad local minima early.
    # But a structured init helps start near a valid solution.
    
    # Let's try a few different initial aspect ratios and pick the best via short SA or just one good one.
    # Given time constraint, we will do one main SA run with random restarts if stuck.
    
    # Strategy: 
    # 1. Start with a valid, non-overlapping configuration (small radii).
    # 2. Anneal: 
    #    - Move positions
    #    - Grow radii
    #    - Adjust aspect ratio
    
    # Initial valid state:
    # Grid of 21 circles in a 1x1 square.
    # 5 columns, 5 rows. 
    # Radius approx 0.15.
    # To be safe, start with r=0.05.
    
    cols = 5
    rows = 5
    # Indices for 21 circles in 5x5 grid (0..20)
    # We fill row by row.
    for i in range(N):
        row = i // cols
        col = i % cols
        cx[i] = (col + 0.5) / cols
        cy[i] = (row + 0.5) / rows
        r[i] = 0.05
        
    current_W = W
    current_H = H
    current_cx = cx.copy()
    current_cy = cy.copy()
    current_r = r.copy()
    
    current_E = _energy(current_cx, current_cy, current_r, current_W, current_H)
    current_val = np.sum(current_r)
    
    best_cx = current_cx.copy()
    best_cy = current_cy.copy()
    best_r = current_r.copy()
    best_W = current_W
    best_H = current_H
    best_E = current_E
    best_val = current_val
    
    # SA Parameters
    T = 0.1  # Initial Temperature
    T_min = 1e-6
    cooling = 0.9995
    
    # Move parameters
    move_pos_scale = 0.05
    move_r_scale = 0.01
    move_W_scale = 0.05
    
    it = 0
    max_iters = 50000 # Upper bound, will stop by time
    
    while time.time() < deadline - 5.0 and T > T_min:
        it += 1
        
        # Choose move type
        move_type = random.random()
        
        new_cx = current_cx.copy()
        new_cy = current_cy.copy()
        new_r = current_r.copy()
        new_W = current_W
        new_H = current_H
        
        if move_type < 0.4:
            # Move a random circle
            idx = random.randint(0, N-1)
            new_cx[idx] += random.gauss(0, move_pos_scale)
            new_cy[idx] += random.gauss(0, move_pos_scale)
        elif move_type < 0.6:
            # Grow/Shrink all radii
            factor = 1.0 + random.gauss(0, move_r_scale)
            factor = max(0.9, min(1.1, factor))
            new_r = current_r * factor
        elif move_type < 0.8:
            # Adjust Aspect Ratio
            new_W = current_W + random.gauss(0, move_W_scale)
            new_W = np.clip(new_W, 0.5, 1.5) # Keep aspect ratio reasonable
            new_H = 2.0 - new_W
        elif move_type < 0.9:
            # Shift all circles in X
            shift = random.gauss(0, move_pos_scale * 0.5)
            new_cx = current_cx + shift
        else:
            # Shift all circles in Y
            shift = random.gauss(0, move_pos_scale * 0.5)
            new_cy = current_cy + shift
            
        # Calculate new energy
        new_E = _energy(new_cx, new_cy, new_r, new_W, new_H)
        
        # Acceptance criterion
        delta_E = new_E - current_E
        if delta_E < 0:
            # Always accept improvement
            current_cx = new_cx
            current_cy = new_cy
            current_r = new_r
            current_W = new_W
            current_H = new_H
            current_E = new_E
        else:
            # Metropolis criterion
            if T > 1e-10:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_cx = new_cx
                    current_cy = new_cy
                    current_r = new_r
                    current_W = new_W
                    current_H = new_H
                    current_E = new_E
                    
        # Update best
        # Only consider as "best" if it's nearly valid (E < 1e-4) or if it has higher value and low E
        # Actually, we want to maximize sum(r) subject to validity.
        # If E is small, it's valid.
        if current_E < 1e-4:
             val = np.sum(current_r)
             if val > best_val:
                 best_val = val
                 best_cx = current_cx.copy()
                 best_cy = current_cy.copy()
                 best_r = current_r.copy()
                 best_W = current_W
                 best_H = current_H
                 best_E = current_E
        elif current_E < best_E:
             # Track best non-valid state to help escape? 
             # Usually better to just track valid bests.
             pass
             
        # Cooling
        T *= cooling
        
        # Adaptive move sizes
        # If we are accepting a lot, increase move size. If rejecting, decrease.
        # Simple heuristic: decrease over time
        if it % 100 == 0:
             move_pos_scale *= 0.99
             move_r_scale *= 0.99
             move_W_scale *= 0.99
             
    # Final Polish: Local search on the best valid state found
    # Try to grow radii further by moving circles
    # Use the best valid state
    if best_E < 1e-4:
        cx, cy, r, W, H = best_cx, best_cy, best_r, best_W, best_H
        # Local refinement
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            # Try growing
            r_trial = r * 1.001
            E_trial = _energy(cx, cy, r_trial, W, H)
            if E_trial < 1e-4:
                r = r_trial
                best_val = np.sum(r)
            else:
                # Try moving a random circle to make room
                idx = random.randint(0, N-1)
                cx_trial = cx.copy()
                cy_trial = cy.copy()
                cx_trial[idx] += random.gauss(0, 0.01)
                cy_trial[idx] += random.gauss(0, 0.01)
                # Project to bounds
                cx_trial[idx] = np.clip(cx_trial[idx], r[idx], W - r[idx])
                cy_trial[idx] = np.clip(cy_trial[idx], r[idx], H - r[idx])
                
                E_move = _energy(cx_trial, cy_trial, r, W, H)
                if E_move < 1e-4:
                    cx, cy = cx_trial, cy_trial
                    # Try growing again
                    r_trial = r * 1.001
                    E_grow = _energy(cx, cy, r_trial, W, H)
                    if E_grow < 1e-4:
                        r = r_trial
                        best_val = np.sum(r)
                        
        # Final check
        if _energy(cx, cy, r, W, H) < 1e-4:
            return np.column_stack((cx, cy, r))
    
    # Fallback: Return the best valid state we found, or a simple grid if none
    if best_E < 1e-4:
        return np.column_stack((best_cx, best_cy, best_r))
    
    # Extreme fallback: Small grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])
