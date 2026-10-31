import numpy as np
import random

N = 21

def construct():
    # Perimeter is 4, so 2*(W+H) = 4 => W+H = 2.
    # We will test a range of aspect ratios (W/H) to find the best packing.
    # For N=21, a square (W=1, H=1) or slightly rectangular is likely optimal.
    
    best_score = -1
    best_result = None
    
    # Define aspect ratios to test
    # W ranges from 0.7 to 1.3 (H = 2-W)
    aspect_ratios = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15]
    
    for w_ratio in aspect_ratios:
        W = w_ratio
        H = 2.0 - W
        
        # Initialize positions using a grid
        # 21 circles. 5x4 grid = 20. Add 1.
        # Let's use a 5x5 grid and remove 4 corners? Or just 5x4 + 1 center?
        # A 5x4 grid (5 cols, 4 rows) has 20 spots.
        # Let's try 4 cols, 5 rows (20 spots) + 1 center.
        
        x_init = []
        y_init = []
        cols, rows = 4, 5
        for i in range(cols):
            for j in range(rows):
                x_init.append((i + 0.5) * W / cols)
                y_init.append((j + 0.5) * H / rows)
        
        if len(x_init) < N:
            x_init.append(W / 2.0)
            y_init.append(H / 2.0)
            
        x = np.array(x_init[:N])
        y = np.array(y_init[:N])
        
        # Vectorized radius calculation
        def calc_radii(x, y, W, H):
            # Wall constraints
            r_wall = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
            
            # Pairwise distances
            # x shape (N,), y shape (N,)
            # dx shape (N, N)
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Distance constraints: r_i <= dist(i, j) / 2
            # We need min over j != i of dist(i, j) / 2
            # Set diagonal to infinity to ignore self-distance
            np.fill_diagonal(dist, np.inf)
            r_neighbor = np.min(dist, axis=1) / 2.0
            
            # Final radius is min of wall and neighbor constraints
            r = np.minimum(r_wall, r_neighbor)
            return r
        
        # Simulated Annealing
        # Temperature schedule
        T_start = 0.5
        T_end = 0.001
        iterations = 5000
        cooling_rate = (T_end / T_start) ** (1.0 / iterations)
        
        x_curr = x.copy()
        y_curr = y.copy()
        r_curr = calc_radii(x_curr, y_curr, W, H)
        score_curr = np.sum(r_curr)
        
        best_local_score = score_curr
        best_local_x = x_curr.copy()
        best_local_y = y_curr.copy()
        
        T = T_start
        
        for _ in range(iterations):
            # Pick a random circle
            k = random.randint(0, N - 1)
            old_x = x_curr[k]
            old_y = y_curr[k]
            
            # Perturb
            # Step size can be adaptive or fixed. Let's use a fixed small step for stability
            step = 0.05
            x_curr[k] += random.uniform(-step, step)
            y_curr[k] += random.uniform(-step, step)
            
            # Clamp to bounds
            x_curr[k] = max(0.0, min(W, x_curr[k]))
            y_curr[k] = max(0.0, min(H, y_curr[k]))
            
            # Calculate new radii and score
            r_new = calc_radii(x_curr, y_curr, W, H)
            score_new = np.sum(r_new)
            
            delta = score_new - score_curr
            
            # Acceptance criterion
            if delta > 0 or random.random() < np.exp(delta / T):
                score_curr = score_new
                if score_new > best_local_score:
                    best_local_score = score_new
                    best_local_x = x_curr.copy()
                    best_local_y = y_curr.copy()
            else:
                # Revert
                x_curr[k] = old_x
                y_curr[k] = old_y
                # score_curr remains unchanged, no need to recalc r_curr unless we want to be precise
                # but since we reverted, state is same as before, so score_curr is valid.
            
            T *= cooling_rate
            
        # Update global best
        if best_local_score > best_score:
            best_score = best_local_score
            best_r = calc_radii(best_local_x, best_local_y, W, H)
            best_result = np.column_stack((best_local_x, best_local_y, best_r))
            
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
