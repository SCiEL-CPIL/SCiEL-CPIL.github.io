import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    best_val = -np.inf
    best_x = None
    
    # Define the penalty function
    def penalty_func(params):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Boundary penalties
        pen = 0.0
        for i in range(N):
            pen += max(0, r[i] - x[i])**2
            pen += max(0, r[i] - (1.0 - x[i]))**2
            pen += max(0, r[i] - y[i])**2
            pen += max(0, r[i] - (1.0 - y[i]))**2
            
        # Overlap penalties
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    pen += overlap**2
        
        # Objective: minimize penalty - alpha * sum(r)
        # We want to maximize sum(r), so subtract a large multiple of sum(r)
        alpha = 100.0
        return pen - alpha * np.sum(r)

    # Helper to check validity
    def is_valid(params, tol=1e-5):
        x = params[0:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Check boundaries
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > 1.0 - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > 1.0 - r[i] + tol:
                return False
                
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    # Generate initial configurations
    inits = []
    
    # 1. Grid 5x5 (take first 21)
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    x_flat = xs.flatten()[:21]
    y_flat = ys.flatten()[:21]
    r_flat = np.full(21, 0.05)
    inits.append(np.concatenate([x_flat, y_flat, r_flat]))
    
    # 2. Hexagonal lattice
    hex_x = []
    hex_y = []
    for row in range(5):
        for col in range(5):
            xx = 0.1 + col * 0.2 + (0.1 if row % 2 else 0)
            yy = 0.1 + row * 0.17
            if xx < 0.9 and yy < 0.9:
                hex_x.append(xx)
                hex_y.append(yy)
    # Shuffle and take 21
    indices = np.random.permutation(len(hex_x))[:21]
    x_h = np.array(hex_x)[indices]
    y_h = np.array(hex_y)[indices]
    r_h = np.full(21, 0.05)
    inits.append(np.concatenate([x_h, y_h, r_h]))
    
    # 3. Random
    np.random.seed(42)
    x_r = np.random.uniform(0.1, 0.9, 21)
    y_r = np.random.uniform(0.1, 0.9, 21)
    r_r = np.full(21, 0.05)
    inits.append(np.concatenate([x_r, y_r, r_r]))
    
    # Add more random restarts if time permits
    # We'll iterate through inits and add more random ones if time is left
    
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(penalty_func, init, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if is_valid(res.x):
                val = np.sum(res.x[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
        
        # Add a new random init if time is available
        if time.time() < deadline - 10:
            x_new = np.random.uniform(0.1, 0.9, 21)
            y_new = np.random.uniform(0.1, 0.9, 21)
            r_new = np.full(21, 0.05)
            inits.append(np.concatenate([x_new, y_new, r_new]))
            
    if best_x is None:
        # Fallback: return the seed-like grid if optimization failed
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Construct output array (x, y, r)
    x = best_x[0:N]
    y = best_x[N:2*N]
    r = best_x[2*N:3*N]
    
    # Final validation and slight shrink if needed to ensure strict validity
    # The evaluator might be strict about "disjoint interiors"
    # We check and shrink if necessary
    def shrink_if_invalid(params, steps=100):
        for _ in range(steps):
            if is_valid(params, tol=1e-6):
                return params
            # Shrink radii by 1%
            params[2*N:3*N] *= 0.99
        return params

    best_x = shrink_if_invalid(best_x)
    
    x = best_x[0:N]
    y = best_x[N:2*N]
    r = best_x[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
