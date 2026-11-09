import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # We assume the optimal rectangle is 1x1 (perimeter 4)
    # If a different aspect ratio is better, we might miss it, but 1x1 is a strong baseline.
    # Let's stick to 1x1 for simplicity and robustness.
    
    def penalty_function(params):
        if time.time() > deadline:
            return np.inf
            
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: Minimize negative sum of radii + penalties
        obj = -np.sum(r)
        
        # Boundary penalties
        # x_i - r_i >= 0 -> violation if x_i < r_i
        # x_i + r_i <= 1 -> violation if x_i > 1 - r_i
        # y_i - r_i >= 0
        # y_i + r_i <= 1
        
        viol_x_min = np.maximum(0, r - x)
        viol_x_max = np.maximum(0, x - (1 - r))
        viol_y_min = np.maximum(0, r - y)
        viol_y_max = np.maximum(0, y - (1 - r))
        
        boundary_pen = np.sum(viol_x_min**2) + np.sum(viol_x_max**2) + \
                       np.sum(viol_y_min**2) + np.sum(viol_y_max**2)
        
        # Overlap penalties
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        return obj + 10.0 * boundary_pen + 10.0 * overlap_pen

    def is_valid(x, y, r):
        # Check boundaries
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return False
        return True

    def get_score(x, y, r):
        return np.sum(r)

    # Run multiple random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initial guess: Random positions and small radii
        # Or use a grid
        if restart == 0:
            # Grid start
            side = int(np.ceil(np.sqrt(N)))
            x = np.array([(i + 0.5) / side for i in range(side) for _ in range(side)][:N])
            y = np.array([_ for _ in range(side) for j in range(side) for _ in range(1)][:N])
            # Correct grid generation
            x = []
            y = []
            for i in range(side):
                for j in range(side):
                    if len(x) < N:
                        x.append((i + 0.5) / side)
                        y.append((j + 0.5) / side)
            x = np.array(x)
            y = np.array(y)
            r = np.full(N, 0.1)
        else:
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.05)
            
        params = np.concatenate([x, y, r])
        
        # Bounds for optimizer
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
        for i in range(N):
            bounds.append((0, 1)) # y
        for i in range(N):
            bounds.append((0, 0.5)) # r (max radius in 1x1 box is 0.5)
            
        try:
            res = minimize(penalty_function, params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.success or res.fun < penalty_function(params):
                x_opt = res.x[0:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Validate
                if is_valid(x_opt, y_opt, r_opt):
                    score = get_score(x_opt, y_opt, r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_opt, y_opt, r_opt))
                        
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to seed-like solution
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
