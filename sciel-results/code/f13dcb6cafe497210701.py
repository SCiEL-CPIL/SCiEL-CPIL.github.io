import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave margin
    
    best_score = 0.0
    best_config = None
    
    # Helper to check validity and get score
    def get_score(config):
        # config is (N, 3)
        xs = config[:, 0]
        ys = config[:, 1]
        rs = config[:, 2]
        
        # Check bounds
        if np.any(rs <= 0) or np.any(rs > 0.5):
            return -np.inf
        if np.any(xs < rs) or np.any(xs > 1 - rs) or np.any(ys < rs) or np.any(ys > 1 - rs):
            return -np.inf
            
        # Check overlaps
        # Vectorized distance check
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum * rad_sum - 1e-9: # Slight tolerance
                    return -np.inf
        
        return np.sum(rs)

    def penalty_objective(config):
        # config is (N, 3)
        # We want to minimize -sum(r) + penalty
        xs = config[:, 0]
        ys = config[:, 1]
        rs = config[:, 2]
        
        # Ensure radii are positive for calculation stability
        rs = np.maximum(rs, 1e-6)
        
        # Boundary penalties
        # x < r or x > 1-r
        pen_x = np.sum(np.maximum(0, rs - xs)**2 + np.maximum(0, xs - (1-rs))**2)
        pen_y = np.sum(np.maximum(0, rs - ys)**2 + np.maximum(0, ys - (1-rs))**2)
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum * rad_sum:
                    diff = rad_sum - np.sqrt(dist_sq)
                    pen_overlap += diff**2
        
        total_penalty = pen_x + pen_y + pen_overlap
        return -np.sum(rs) + 100.0 * total_penalty

    # Initializations
    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells[:N])

    def init_random():
        xs = np.random.rand(N)
        ys = np.random.rand(N)
        rs = np.random.rand(N) * 0.05 + 0.01
        return np.column_stack((xs, ys, rs))

    def init_hex():
        # Hexagonal packing approximation
        # ... implement if needed, random is often better with optimizer
        pass
        
    # Optimization loop
    starts = [init_grid(), init_random(), init_random()]
    
    for start in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(penalty_objective, start, method='L-BFGS-B', 
                           bounds=[(0,1)]*N + [(0,1)]*N + [(0,0.5)]*N,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < 0: # Heuristic check
                config = res.x
                score = get_score(config)
                if score > best_score:
                    best_score = score
                    best_config = config
        except Exception as e:
            pass
            
    # If no valid config found, return seed
    if best_config is None:
        return init_grid()
        
    return best_config
