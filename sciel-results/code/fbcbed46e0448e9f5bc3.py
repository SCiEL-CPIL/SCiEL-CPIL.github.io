import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145

def _min_area(points):
    # Vectorized calculation of min area
    # Points shape (N, 2)
    # Compute all pairs? No, just triplets.
    # N=11 is small, loop is fine.
    min_a = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                # Area = 0.5 * |cross|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = abs(v1[0]*v2[1] - v1[1]*v2[0]) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a

def _penalized_min_area(flat_params):
    points = flat_params.reshape(-1, 2)
    # Penalty for outside triangle
    penalty = 0.0
    for p in points:
        x, y = p
        # y < 0
        if y < 0: penalty += -y * 100
        # y > S3H
        if y > S3H: penalty += (y - S3H) * 100
        # x < 0
        if x < 0: penalty += -x * 100
        # x > 1
        if x > 1: penalty += (x - 1) * 100
        # Left edge: y > sqrt(3)*x
        if y > np.sqrt(3)*x: penalty += (y - np.sqrt(3)*x) * 100
        # Right edge: y > sqrt(3)*(1-x)
        if y > np.sqrt(3)*(1-x): penalty += (y - np.sqrt(3)*(1-x)) * 100
        
    ma = _min_area(points)
    return -ma + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Symmetric
    # Vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, S3H])
    
    # Midpoints
    m1 = (v1 + v2) / 2.0
    m2 = (v2 + v3) / 2.0
    m3 = (v3 + v1) / 2.0
    
    # Center
    c = (v1 + v2 + v3) / 3.0
    
    # We need 11 points. We have 3+3+1 = 7. Need 4 more.
    # Place them in a ring around the center?
    # Or use a grid.
    
    # Let's create a list of candidate points and pick 11
    # Or just use random symmetric initialization.
    
    # Better: Use a set of known good structures.
    # Structure:
    # Row 0 (bottom): 4 points evenly spaced?
    # Row 1: 3 points?
    # Row 2: 2 points?
    # Row 3: 1 point?
    # Total 10. Add 1 in center?
    
    # Let's just use DE with a good initial population.
    
    bounds = [(0, 1), (0, S3H)] * N
    
    best_flat = None
    best_val = float('inf') # We minimize -area + penalty, so we want low value.
    # Wait, _penalized_min_area returns -area + penalty.
    # We want to MAXIMIZE area. So we MINIMIZE -area.
    # So we track the minimum value of _penalized_min_area.
    
    # Run multiple DE restarts
    seeds = [42, 123, 456, 789, 321]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        try:
            result = differential_evolution(
                _penalized_min_area,
                bounds,
                maxiter=50,
                popsize=20,
                seed=seed,
                tol=1e-5,
                polish=False, # We will polish manually
                init='latinhypercube'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_flat = result.x.copy()
                
        except Exception as e:
            pass
            
    # Local Polish
    if best_flat is not None:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            try:
                # Nelder-Mead polish
                result = minimize(
                    _penalized_min_area,
                    best_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_flat = result.x.copy()
            except Exception:
                pass
            
        # Second polish with L-BFGS-B (might be faster/stable)
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 2:
            try:
                result = minimize(
                    _penalized_min_area,
                    best_flat,
                    method='L-BFGS-B',
                    bounds=bounds, # L-BFGS-B supports bounds
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_flat = result.x.copy()
            except Exception:
                pass

    points = best_flat.reshape(-1, 2)
    
    # Final Projection to ensure validity
    for i in range(len(points)):
        x, y = points[i]
        y = max(0, min(S3H, y))
        x_min = y / np.sqrt(3)
        x_max = 1 - y / np.sqrt(3)
        x = max(x_min, min(x_max, x))
        points[i] = [x, y]
        
    return points
