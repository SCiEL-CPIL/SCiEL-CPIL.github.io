"""Circle packing in perimeter-4 rectangle using force-directed relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_value = 0.0
    best_result = None
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Generate random initial positions and radii
        rng = np.random.default_rng(attempt * 42 + 7)
        
        # Start with a grid-like arrangement as initial guess
        side = 5  # 5x5 grid
        centers = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    x = (i + 0.5) / side + rng.uniform(-0.05, 0.05)
                    y = (j + 0.5) / side + rng.uniform(-0.05, 0.05)
                    centers.append([x, y])
                    radii.append(0.1)
        
        centers = np.array(centers[:N])
        radii = np.array(radii[:N])
        
        # Optimize using force-directed relaxation
        result = _force_directed_optimize(centers, radii, deadline - (time.time() - start_time))
        if result is not None:
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_result = result
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _force_directed_optimize(centers, radii, time_budget):
    """Force-directed relaxation to pack circles in a rectangle with perimeter 4."""
    start_time = time.time()
    
    n = len(centers)
    # Rectangle with perimeter 4: 2*(w+h) = 4, so w+h = 2
    # We'll optimize w, h, and circle positions/radii
    # Let's fix aspect ratio initially, then optimize
    
    # Parameter vector: [w, h, x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    # Constraint: 2*(w+h) <= 4, i.e., w+h <= 2
    # Also: circles must be inside rectangle and non-overlapping
    
    def objective(params):
        w = params[0]
        h = params[1]
        if w <= 0 or h <= 0:
            return 1e6
        
        # Enforce perimeter constraint softly
        if w + h > 2.0:
            return 1e6 + (w + h - 2.0) * 100
        
        centers_opt = params[2:2+2*n].reshape(n, 2)
        radii_opt = params[2+2*n:2+3*n]
        
        # Penalty for circles outside rectangle
        penalty = 0.0
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if x - r < 0:
                penalty += (x - r) * 100
            if x + r > w:
                penalty += (x + r - w) * 100
            if y - r < 0:
                penalty += (y - r) * 100
            if y + r > h:
                penalty += (y + r - h) * 100
        
        # Penalty for overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii_opt[i] + radii_opt[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) * 50
        
        # Maximize sum of radii (negate for minimization)
        return -np.sum(radii_opt) + penalty
    
    # Initial parameters
    w_init = 1.0
    h_init = 1.0
    params_init = np.concatenate(([w_init, h_init], centers.flatten(), radii))
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]  # w, h
    for i in range(n):
        bounds.append((0, 2))  # x
        bounds.append((0, 2))  # y
        bounds.append((0.01, 0.5))  # r
    
    # Optimize
    try:
        res = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e4:
            w = res.x[0]
            h = res.x[1]
            if w + h <= 2.0 + 1e-6 and w > 0 and h > 0:
                centers_opt = res.x[2:2+2*n].reshape(n, 2)
                radii_opt = res.x[2+2*n:2+3*n]
                
                # Verify solution is valid
                valid = True
                for i in range(n):
                    x, y = centers_opt[i]
                    r = radii_opt[i]
                    if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                        valid = False
                        break
                    if r < 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = centers_opt[i, 0] - centers_opt[j, 0]
                            dy = centers_opt[i, 1] - centers_opt[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < radii_opt[i] + radii_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    # Scale to ensure perimeter <= 4
                    scale = min(1.0, 2.0 / (w + h))
                    centers_final = centers_opt * scale
                    radii_final = radii_opt * scale
                    w_final = w * scale
                    h_final = h * scale
                    
                    result = np.column_stack([centers_final, radii_final])
                    return result
    except Exception:
        pass
    
    return None
