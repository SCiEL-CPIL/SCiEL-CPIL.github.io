"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_config = None
    
    # Helper to check validity and get sum
    def get_score(config, W, H):
        # config: (N, 3) array of [x, y, r]
        if len(config) != N:
            return 0.0, False
        # Check boundaries
        for i in range(N):
            x, y, r = config[i]
            if r < 0:
                return 0.0, False
            if x < r - 1e-6 or x > W - r + 1e-6:
                return 0.0, False
            if y < r - 1e-6 or y > H - r + 1e-6:
                return 0.0, False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1 + r2) - 1e-6:
                    return 0.0, False
        
        s = np.sum(config[:, 2])
        return s, True

    # Objective function for optimizer
    def objective(vars, W, H):
        # vars: flat array of x, y, r for each circle
        config = vars.reshape(N, 3)
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            x, y, r = config[i]
            if r < 0:
                penalty += (r) * 100
            # x bounds
            if x < r:
                penalty += (r - x) * 10
            if x > W - r:
                penalty += (x - (W - r)) * 10
            # y bounds
            if y < r:
                penalty += (r - y) * 10
            if y > H - r:
                penalty += (y - (H - r)) * 10
        
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                min_dist = r1 + r2
                if dist < min_dist:
                    penalty += (min_dist - dist) * 10
        
        sum_r = np.sum(config[:, 2])
        return -sum_r + penalty

    # Initialize grid
    def init_grid(W, H):
        # Try to find best N x M grid
        best_init = None
        best_init_score = -1
        
        for M in range(1, 8):
            for N_cols in range(1, 8):
                if N_cols * M < N:
                    continue
                # Aspect ratio match
                ar_rect = W / H
                ar_grid = N_cols / M
                if abs(ar_rect - ar_grid) > 1.5:
                    continue
                
                # Calculate max radius for this grid
                # W = N_cols * 2r, H = M * 2r
                r_w = W / (2 * N_cols)
                r_h = H / (2 * M)
                r = min(r_w, r_h)
                
                # Create config
                config = np.zeros((N, 3))
                idx = 0
                for j in range(M):
                    for i in range(N_cols):
                        if idx >= N:
                            break
                        x = r + i * 2 * r
                        y = r + j * 2 * r
                        config[idx] = [x, y, r]
                        idx += 1
                    if idx >= N:
                        break
                
                if idx < N:
                    # Not enough spots, try to squeeze in or skip this grid
                    continue
                
                # Score: sum of radii
                s = np.sum(config[:, 2])
                if s > best_init_score:
                    best_init_score = s
                    best_init = config.copy()
        
        if best_init is None:
            # Fallback: simple square grid
            side = int(np.ceil(np.sqrt(N)))
            r = min(W, H) / (2 * side) * 0.9
            config = np.zeros((N, 3))
            idx = 0
            for j in range(side):
                for i in range(side):
                    if idx >= N:
                        break
                    x = r + i * 2 * r
                    y = r + j * 2 * r
                    config[idx] = [x, y, r]
                    idx += 1
            best_init = config
        return best_init

    # Loop over aspect ratios
    W_values = np.linspace(0.5, 1.5, 20)
    
    for W in W_values:
        if time.time() - start_time > time_limit:
            break
            
        H = 2.0 - W
        if H <= 0:
            continue
        
        init_config = init_grid(W, H)
        x0 = init_config.flatten()
        
        # Run optimization
        try:
            res = minimize(objective, x0, args=(W, H), method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.success or res.fun < 0:
                config = res.x.reshape(N, 3)
                # Ensure radii are non-negative
                config[:, 2] = np.maximum(config[:, 2], 0)
                
                s, valid = get_score(config, W, H)
                if valid and s > best_sum:
                    best_sum = s
                    best_config = config.copy()
        except Exception:
            continue

    if best_config is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_config
# EVOLVE-BLOCK-END
