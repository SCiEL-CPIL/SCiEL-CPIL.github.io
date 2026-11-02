"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        # x is [x1,y1,r1, x2,y2,r2, ...]
        valid = True
        # Check bounds
        for i in range(N):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            if ri < 1e-9 or xi - ri < -1e-6 or xi + ri > w + 1e-6 or yi - ri < -1e-6 or yi + ri > h + 1e-6:
                valid = False
                break
        if not valid:
            return
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < (min_dist - 1e-7)**2:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            val = np.sum(x[2::3])
            if val > best_value:
                best_value = val
                best_result = x.copy()

    # Loop over aspect ratios (widths)
    # w + h = 2, so h = 2 - w
    # Try a range of widths
    widths = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize x0 with a grid
        # Estimate grid size
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        x0 = np.zeros(3*N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3*i] = (col + 0.5) * w / a
            x0[3*i+1] = (row + 0.5) * h / b
            x0[3*i+2] = min(w/a, h/b) * 0.45 # Initial radius guess
            
        # Objective function for fixed w, h
        def objective(params):
            # params: [x1, y1, r1, ..., x21, y21, r21]
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            # L-BFGS-B enforces bounds, so we assume bounds are respected for x, y, r
            # But we check r > 0 just in case
            if np.any(rs <= 0):
                return 1e10 + np.sum(np.minimum(rs, 0)**2)
                
            # Circle-Circle Overlap Penalty
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    if dist < min_dist:
                        # Quadratic penalty for overlap
                        penalty += (min_dist - dist)**2
            
            # We want to maximize sum of radii, so minimize -sum(r) + penalty
            return penalty - np.sum(rs)
        
        # Bounds for L-BFGS-B
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline:
                break
            # Check if the result is valid and better
            # L-BFGS-B might return a point that violates constraints slightly due to numerical issues?
            # No, it projects onto bounds. But circle-circle is a penalty, so it might not be zero.
            # We rely on check_and_update to verify strict validity.
            check_and_update(res.x, w, h)
        except Exception:
            pass

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3*N)
        for i in range(a):
            for j in range(b):
                idx = i*b + j
                best_result[3*idx] = (i + 0.5) * w / a
                best_result[3*idx+1] = (j + 0.5) * h / b
                best_result[3*idx+2] = r
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3*i:3*i+3]
    
    return result
# EVOLVE-BLOCK-END
