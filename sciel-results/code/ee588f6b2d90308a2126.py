import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0
    best_config = None
    
    # Try a few aspect ratios
    aspect_ratios = [1.0, 1.5, 2.0, 0.75]
    
    for rho in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Rectangle dimensions
        w = 2 * rho / (1 + rho)
        h = 2 / (1 + rho)
        
        # Objective function to minimize
        # We want to maximize sum(r_i), so minimize -sum(r_i)
        # Penalty for overlaps and out-of-bounds
        def objective(vars):
            # vars: [x1, y1, r1, ..., xN, yN, rN]
            x = vars[0::3]
            y = vars[1::3]
            r = vars[2::3]
            
            # Sum of radii
            val = -np.sum(r)
            
            # Penalty for out of bounds
            # x_i - r_i >= 0, x_i + r_i <= w
            # y_i - r_i >= 0, y_i + r_i <= h
            pen = 0.0
            for i in range(N):
                if x[i] - r[i] < 0:
                    pen += (x[i] - r[i])**2
                if x[i] + r[i] > w:
                    pen += (x[i] + r[i] - w)**2
                if y[i] - r[i] < 0:
                    pen += (y[i] - r[i])**2
                if y[i] + r[i] > h:
                    pen += (y[i] + r[i] - h)**2
            
            # Penalty for overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                    min_dist = r[i] + r[j]
                    if dist_sq < min_dist**2:
                        # Overlap penalty
                        overlap = min_dist - np.sqrt(dist_sq)
                        pen += overlap**2
            
            return val + 100.0 * pen
        
        # Initial guess: Grid
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(3*N)
        for i in range(N):
            ix = i % side
            iy = i // side
            x0[3*i] = (ix + 0.5) * w / side
            x0[3*i+1] = (iy + 0.5) * h / side
            x0[3*i+2] = min(w, h) / (2 * side) * 0.9 # Initial radius
            
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w)) # x
            bounds.append((0, h)) # y
            bounds.append((1e-6, min(w, h)/2)) # r
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-6})
            
            # Check if valid
            x = res.x[0::3]
            y = res.x[1::3]
            r = res.x[2::3]
            
            # Verify validity
            valid = True
            for i in range(N):
                if x[i] - r[i] < -1e-5 or x[i] + r[i] > w + 1e-5 or \
                   y[i] - r[i] < -1e-5 or y[i] + r[i] > h + 1e-5:
                    valid = False
                    break
                if not valid:
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                        if dist_sq < (r[i]+r[j])**2 - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_config = np.column_stack((x, y, r))
                    
        except Exception:
            pass
            
    if best_config is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
