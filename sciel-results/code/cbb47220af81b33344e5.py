import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    best_r = 0.0
    best_config = None
    
    # Test a range of aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2, 1.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
            
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        r_max_possible = min(w, h) / 2
        
        # Binary search for r
        lo = 0.1
        hi = r_max_possible
        
        for _ in range(15): # 15 steps of binary search
            if time.time() - start_time > time_limit:
                break
                
            mid = (lo + hi) / 2
            
            # Try to pack N circles of radius mid
            success, config = try_pack(mid, w, h)
            
            if success:
                lo = mid
                best_r = mid
                best_config = config
            else:
                hi = mid
                
    if best_config is not None:
        result = np.column_stack((best_config[:, 0], best_config[:, 1], np.full(N, best_r)))
        return result
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

def try_pack(r, w, h, max_iters=1000, restarts=5):
    # Try to pack N circles of radius r in w x h
    # Return (success, config)
    
    for _ in range(restarts):
        # Random initialization
        X = np.random.uniform(r, w - r, N)
        Y = np.random.uniform(r, h - r, N)
        
        # Relaxation
        step_size = 0.05
        decay = 0.99
        
        for i in range(max_iters):
            # Calculate forces
            Fx = np.zeros(N)
            Fy = np.zeros(N)
            
            # Pairwise repulsion
            dx = X[:, np.newaxis] - X[np.newaxis, :]
            dy = Y[:, np.newaxis] - Y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            
            overlap = np.maximum(0, 2*r - dist)
            
            inv_dist = np.where(dist > 0, 1/dist, 0)
            Fx += np.sum(overlap * dx * inv_dist, axis=1)
            Fy += np.sum(overlap * dy * inv_dist, axis=1)
            
            # Wall forces
            Fx += np.where(X < r, (r - X) / r, 0)
            Fx += np.where(X > w - r, (X - (w - r)) / r, 0)
            Fy += np.where(Y < r, (r - Y) / r, 0)
            Fy += np.where(Y > h - r, (Y - (h - r)) / r, 0)
            
            # Update positions
            X += step_size * Fx
            Y += step_size * Fy
            
            # Clip to bounds
            X = np.clip(X, r, w - r)
            Y = np.clip(Y, r, h - r)
            
            step_size *= decay
            
            # Check if valid
            dx = X[:, np.newaxis] - X[np.newaxis, :]
            dy = Y[:, np.newaxis] - Y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dist, np.inf)
            min_dist = np.min(dist)
            
            if min_dist >= 2*r - 1e-5:
                if np.all(X >= r - 1e-5) and np.all(X <= w - r + 1e-5) and np.all(Y >= r - 1e-5) and np.all(Y <= h - r + 1e-5):
                    return True, np.column_stack((X, Y))
        
    return False, None
