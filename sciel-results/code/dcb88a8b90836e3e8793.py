import numpy as np
import random

N = 21

def construct():
    # We will try a few different aspect ratios and initializations
    # and keep the best result.
    best_score = -1
    best_result = None
    
    # Perimeter is 4, so 2*(W+H) = 4 => W+H = 2.
    # We can vary W from 0.8 to 1.2 (H = 2 - W)
    
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3]
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Initialize with a grid
        # 21 circles. 5x4 grid = 20, add 1. Or 4x5.
        # Let's use a 5x4 grid (5 cols, 4 rows) -> 20 spots. Add 1 in center?
        # Or just 21 spots in a 5x5 grid skipping corners?
        # Simple grid:
        cols = 5
        rows = 4
        # 20 circles in 5x4, 1 extra.
        # Let's just use a 5x5 grid and take first 21?
        # Better: 5x4 grid + 1 in the middle of one of the rows?
        # Let's try 4x5 (4 cols, 5 rows) = 20.
        
        # Let's use a standard grid initialization
        x = np.zeros(N)
        y = np.zeros(N)
        
        # 5 columns, 4 rows = 20 circles.
        # If N=21, we need one more.
        # Let's try 5x5 grid, take 21.
        # Or 4x5 grid, take 20, add 1.
        
        # Let's use a 5x4 grid for 20, and place the 21st in the center of the whole rectangle?
        # No, that might overlap.
        
        # Let's use a 5x5 grid, but only fill 21 positions.
        # Positions:
        # (0,0) (0,1) (0,2) (0,3) (0,4)
        # (1,0) (1,1) (1,2) (1,3) (1,4)
        # (2,0) (2,1) (2,2) (2,3) (2,4)
        # (3,0) (3,1) (3,2) (3,3) (3,4)
        # (4,0) (4,1) (4,2) (4,3) (4,4)
        
        # This is 25 spots. We pick 21.
        # To maximize radii, we want them spread out.
        # Let's just use 5x4 grid (20 circles) and put the 21st in the center of the rectangle?
        # Center is (W/2, H/2).
        # In a 5x4 grid, the center is between (1,2) and (2,2) etc.
        
        # Let's try 4x5 grid (4 cols, 5 rows).
        cols = 4
        rows = 5
        # 20 circles.
        
        x_init = []
        y_init = []
        for i in range(cols):
            for j in range(rows):
                if len(x_init) >= N:
                    break
                # Center of cell (i, j)
                x_init.append((i + 0.5) * W / cols)
                y_init.append((j + 0.5) * H / rows)
        
        # If we have fewer than N, add more?
        # 4*5=20. We need 21.
        # Add one in the center?
        if len(x_init) < N:
            x_init.append(W/2)
            y_init.append(H/2)
            
        x = np.array(x_init[:N])
        y = np.array(y_init[:N])
        
        # Local Search
        # We will perform iterations
        # Calculate radii
        def calc_radii(x, y, W, H):
            r = np.zeros(N)
            for i in range(N):
                # Wall constraints
                r_i = min(x[i], W - x[i], y[i], H - y[i])
                # Neighbor constraints
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    r_i = min(r_i, d / 2.0)
                r[i] = r_i
            return r
        
        # Initial radii
        r = calc_radii(x, y, W, H)
        current_score = np.sum(r)
        
        # Optimization loop
        # Use random restarts or just hill climbing
        # To be robust, we'll do a few rounds of hill climbing
        
        for _ in range(5): # 5 restarts per aspect ratio
            # Randomize positions slightly around grid or fully random?
            # Grid is a good start. Let's perturb.
            x_curr = x.copy()
            y_curr = y.copy()
            r_curr = calc_radii(x_curr, y_curr, W, H)
            score_curr = np.sum(r_curr)
            
            best_local_score = score_curr
            best_local_x = x_curr.copy()
            best_local_y = y_curr.copy()
            
            step_size = 0.05
            for it in range(2000):
                # Pick random circle
                k = random.randint(0, N-1)
                old_x = x_curr[k]
                old_y = y_curr[k]
                
                # Perturb
                x_curr[k] += random.uniform(-step_size, step_size)
                y_curr[k] += random.uniform(-step_size, step_size)
                
                # Clamp to bounds (radii will be 0 if outside, but let's keep them inside)
                x_curr[k] = max(0.0, min(W, x_curr[k]))
                y_curr[k] = max(0.0, min(H, y_curr[k]))
                
                # Recalculate radii
                r_new = calc_radii(x_curr, y_curr, W, H)
                score_new = np.sum(r_new)
                
                if score_new > score_curr:
                    score_curr = score_new
                    if score_new > best_local_score:
                        best_local_score = score_new
                        best_local_x = x_curr.copy()
                        best_local_y = y_curr.copy()
                else:
                    # Revert
                    x_curr[k] = old_x
                    y_curr[k] = old_y
                    # r_curr is unchanged, score_curr unchanged
                
                # Decay step size
                step_size *= 0.9999
                
            # After local search, update global best
            if best_local_score > best_score:
                best_score = best_local_score
                best_result = np.column_stack((best_local_x, best_local_y, calc_radii(best_local_x, best_local_y, W, H)))
                
    # Ensure perimeter constraint is met exactly or less
    # The radii are calculated based on W, H where W+H=2.
    # So perimeter is 4.
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
