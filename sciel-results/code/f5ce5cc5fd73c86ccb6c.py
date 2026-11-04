import numpy as np
import random
import time

N = 26

def construct():
    # Simulated Annealing Parameters
    T_init = 0.1
    T_min = 1e-6
    cooling_rate = 0.999
    max_steps = 50000
    num_restarts = 5
    
    best_score = -np.inf
    best_config = None
    
    # Helper to calculate fitness
    def get_fitness(x, y, r):
        # Sum of radii
        score = np.sum(r)
        
        # Penalty for boundary
        # x < r or x > 1-r
        # y < r or y > 1-r
        # Use soft penalty: square of violation
        pen = 0.0
        # Vectorized boundary check
        # Left/Right
        viol_x = np.maximum(r - x, 0) + np.maximum(x - (1 - r), 0)
        # Bottom/Top
        viol_y = np.maximum(r - y, 0) + np.maximum(y - (1 - r), 0)
        pen += np.sum(viol_x**2) + np.sum(viol_y**2)
        
        # Overlap penalty
        # Calculate all pairwise distances
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = x_diff**2 + y_diff**2
        min_dist = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap is max(0, min_dist - dist)
        # We only care about i < j, but matrix is symmetric.
        # dist is 0 on diagonal, min_dist is 2*r. 
        # We should ignore diagonal.
        # Set diagonal of dist_sq to inf to avoid overlap penalty on self
        np.fill_diagonal(dist_sq, np.inf)
        
        dist = np.sqrt(dist_sq)
        overlap = np.maximum(0, min_dist - dist)
        # Sum of squared overlaps
        pen += np.sum(overlap**2)
        
        # Weighted penalty
        # Large weight to ensure validity
        return score - 100.0 * pen

    for restart in range(num_restarts):
        # Initialize with a grid
        # 5 rows, 6 cols -> 30 cells, pick 26
        # Or 26 random points in a grid-like structure
        # Let's use a jittered grid
        side = int(np.ceil(np.sqrt(N)))
        # Create grid points
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        
        # Generate candidate points
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([xs[i], ys[j]])
        
        # Shuffle and pick N
        random.shuffle(pts)
        x = np.array([p[0] for p in pts[:N]])
        y = np.array([p[1] for p in pts[:N]])
        
        # Initial radii: small enough to fit
        # Max radius in grid cell is 0.5/side
        # Add some noise
        r = np.full(N, 0.1) # Start with uniform radius
        
        # Jitter positions
        x += np.random.normal(0, 0.05, N)
        y += np.random.normal(0, 0.05, N)
        
        # Clamp positions
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0.01, 0.5)
        
        current_score = get_fitness(x, y, r)
        current_x, current_y, current_r = x.copy(), y.copy(), r.copy()
        
        T = T_init
        step = 0
        stall_count = 0
        best_local_score = current_score
        best_local_x, best_local_y, best_local_r = current_x.copy(), current_y.copy(), current_r.copy()
        
        while T > T_min and step < max_steps:
            # Pick a random circle
            k = np.random.randint(N)
            
            # Propose move
            # Move position
            dx = np.random.normal(0, 0.02)
            dy = np.random.normal(0, 0.02)
            # Move radius
            dr = np.random.normal(0, 0.01)
            
            new_x = current_x.copy()
            new_y = current_y.copy()
            new_r = current_r.copy()
            
            new_x[k] += dx
            new_y[k] += dy
            new_r[k] += dr
            
            # Clamp
            new_x[k] = np.clip(new_x[k], 0, 1)
            new_y[k] = np.clip(new_y[k], 0, 1)
            new_r[k] = np.clip(new_r[k], 0.01, 0.5)
            
            new_score = get_fitness(new_x, new_y, new_r)
            
            delta = new_score - current_score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_x, current_y, current_r = new_x, new_y, new_r
                current_score = new_score
                
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_x, best_local_y, best_local_r = current_x.copy(), current_y.copy(), current_r.copy()
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1
                
            # Cool down
            T *= cooling_rate
            step += 1
            
            # Early stop if stalled too much
            if stall_count > 5000:
                break
        
        if best_local_score > best_score:
            best_score = best_local_score
            best_config = (best_local_x, best_local_y, best_local_r)
            
    x, y, r = best_config
    
    # Final validity check
    valid = True
    for i in range(N):
        if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return np.column_stack((x, y, r))
    else:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r_val = 0.5 / side - 0.01
                    cells.append([cx, cy, r_val])
        return np.array(cells[:N])
