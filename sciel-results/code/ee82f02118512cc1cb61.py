import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 26
DEADLINE = 110.0
EPS = 1e-6

def _elapsed(start):
    return time.time() - start

def _compute_radii(x, y, max_iters=50):
    """
    Compute the maximum feasible radii for given center positions.
    Uses iterative relaxation to satisfy r_i + r_j <= d_ij and r_i <= d_bound.
    """
    r = np.zeros(N)
    # Initial radii based on boundary
    for i in range(N):
        r[i] = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
    
    # Precompute distances
    # d[i, j] is distance between i and j
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            d[i, j] = dist
            d[j, i] = dist
            
    for _ in range(max_iters):
        changed = False
        # Boundary check
        for i in range(N):
            max_r_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            if r[i] > max_r_bound:
                r[i] = max_r_bound
                changed = True
        
        # Overlap check
        for i in range(N):
            for j in range(i+1, N):
                dist = d[i, j]
                sum_r = r[i] + r[j]
                if sum_r > dist - EPS:
                    # Need to reduce sum_r to dist - EPS
                    excess = sum_r - (dist - EPS)
                    # Reduce the one with larger radius, or split if equal
                    if r[i] > r[j]:
                        reduction = min(excess, r[i] - r[j])
                        r[i] -= reduction
                        excess -= reduction
                        if excess > 0:
                            # Split the rest
                            r[i] -= excess / 2.0
                            r[j] -= excess / 2.0
                    elif r[j] > r[i]:
                        reduction = min(excess, r[j] - r[i])
                        r[j] -= reduction
                        excess -= reduction
                        if excess > 0:
                            r[i] -= excess / 2.0
                            r[j] -= excess / 2.0
                    else:
                        r[i] -= excess / 2.0
                        r[j] -= excess / 2.0
                    
                    # Ensure non-negative
                    r[i] = max(0.0, r[i])
                    r[j] = max(0.0, r[j])
                    changed = True
        
        if not changed:
            break
            
    return r

def _objective(positions):
    """
    Objective function for the optimizer.
    Minimize negative sum of radii.
    """
    x = positions[:N]
    y = positions[N:]
    
    # Basic bounds check (penalty if out of bounds, though optimizer should handle it)
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return 1000.0
        
    r = _compute_radii(x, y)
    return -np.sum(r)

def _validate_final(x, y, r, eps=1e-5):
    """
    Final strict validation. Returns True if valid, else modifies r to be valid.
    """
    # Boundary
    for i in range(N):
        max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        if r[i] > max_r - eps:
            r[i] = max(0.0, max_r - eps)
            
    # Overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if r[i] + r[j] > dist - eps:
                # Shrink both to fit
                total_r = r[i] + r[j]
                scale = (dist - eps) / total_r if total_r > 0 else 0
                r[i] *= scale
                r[j] *= scale
                # Re-check boundary after shrink (usually fine, but safe)
                max_r_i = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > max_r_i - eps:
                    r[i] = max(0.0, max_r_i - eps)
                max_r_j = min(x[j], 1.0 - x[j], y[j], 1.0 - y[j])
                if r[j] > max_r_j - eps:
                    r[j] = max(0.0, max_r_j - eps)
                    
    return x, y, r

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_val = -np.inf
    best_result = None
    
    # Define bounds for optimizer
    bounds = [(0.0, 1.0) for _ in range(2*N)]
    
    # Multiple initializations
    inits = []
    
    # 1. Grid
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)][:N]
    pos = np.array([p[0] for p in pts] + [p[1] for p in pts])
    inits.append(pos)
    
    # 2. Jittered Grid
    pos2 = pos.copy()
    pos2 += np.random.uniform(-0.05, 0.05, 2*N)
    pos2 = np.clip(pos2, 0.05, 0.95)
    inits.append(pos2)
    
    # 3. Random
    pos3 = np.random.uniform(0.1, 0.9, 2*N)
    inits.append(pos3)
    
    # 4. Corners + Center
    pts4 = []
    pts4.append((0.2, 0.2))
    pts4.append((0.8, 0.2))
    pts4.append((0.2, 0.8))
    pts4.append((0.8, 0.8))
    pts4.append((0.5, 0.5))
    # Fill rest randomly
    while len(pts4) < N:
        pts4.append((np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
    pos4 = np.array([p[0] for p in pts4] + [p[1] for p in pts4])
    inits.append(pos4)
    
    for init_pos in inits:
        if _elapsed(start_time) > DEADLINE - 10.0:
            break
            
        # Run optimization
        try:
            result = minimize(_objective, init_pos, method='Nelder-Mead', 
                              bounds=bounds,
                              options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5})
            
            if result.success or _elapsed(start_time) < DEADLINE - 5.0:
                x = result.x[:N]
                y = result.x[N:]
                
                # Clip to bounds
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                
                r = _compute_radii(x, y)
                
                # Final validation
                x, y, r = _validate_final(x, y, r)
                
                val = np.sum(r)
                
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack([x, y, r])
                    
        except Exception:
            continue
            
    # Fallback
    if best_result is None:
        # Simple grid fallback
        cols, rows = 5, 5
        xs = (np.arange(cols) + 0.5) / cols
        ys = (np.arange(rows) + 0.5) / rows
        pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)][:N]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = _compute_radii(x, y)
        x, y, r = _validate_final(x, y, r)
        best_result = np.column_stack([x, y, r])
        
    return best_result
