import numpy as np
import time
from scipy.optimize import minimize

N = 26
START_TIME = time.time()
TIME_LIMIT = 100.0 # Leave 20s buffer

def construct():
    best_score = -np.inf
    best_result = None
    
    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    
    # Multi-start
    # We will try a few different random perturbations of the grid
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Initialize positions
        if r == 0:
            # Use the grid
            init_pos = np.array(cells[:N])
        else:
            # Jitter the grid
            init_pos = np.array(cells[:N])
            jitter = np.random.uniform(-0.05, 0.05, size=(N, 2))
            init_pos = init_pos + jitter
            # Ensure within [0,1]
            init_pos = np.clip(init_pos, 0.05, 0.95)
            
        # Initialize radii
        # Start with small radii to avoid immediate massive overlaps
        init_rad = np.full(N, 0.05)
        
        # Flatten variables: x0, y0, r0, x1, y1, r1, ...
        x0 = np.concatenate([init_pos[:, 0], init_pos[:, 1], init_rad])
        
        # Define objective
        def objective(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            
            # Sum of radii
            sum_r = np.sum(r)
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist)**2
            
            # Penalty for bounds
            # x + r >= 0, x + r <= 1 => r + x >= 0, r - x <= 1, r + y >= 0, r - y <= 1
            # Let's use simple penalties for being outside [0,1]
            # If x < r, it's sticking out left. If x > 1-r, sticking out right.
            for i in range(N):
                if x[i] < r[i]:
                    penalty += (r[i] - x[i])**2
                if x[i] > 1.0 - r[i]:
                    penalty += (x[i] - (1.0 - r[i]))**2
                if y[i] < r[i]:
                    penalty += (r[i] - y[i])**2
                if y[i] > 1.0 - r[i]:
                    penalty += (y[i] - (1.0 - r[i]))**2
            
            return -sum_r + 10.0 * penalty

        # Bounds for optimizer (loose)
        bounds = [(-1.0, 2.0)] * (3 * N)
        
        # Optimize
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success or (res.fun < best_score if best_result is None else res.fun < best_result[0]):
            # Extract result
            x = res.x[:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            
            # Check validity
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-9 or x[i] > 1.0 - r[i] + 1e-9:
                    valid = False
                    break
                if y[i] < r[i] - 1e-9 or y[i] > 1.0 - r[i] + 1e-9:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = (res.fun, np.column_stack((x, y, r)))
    
    if best_result is None:
        # Fallback to grid if optimization failed completely
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result[1]
