"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _check_valid(x, y, r, tol=1e-7):
    """Check non-overlap and perimeter constraint."""
    n = len(r)
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < r[i] + r[j] - tol:
                return False
    # Perimeter
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    bw = x_max - x_min
    bh = y_max - y_min
    if 2 * (bw + bh) > 4 + tol:
        return False
    return True

def _solve_convex_pack(w, h, x0, y0, r0, max_time=30.0, start_time=None):
    """
    Solve the convex optimization problem for fixed container w, h.
    Variables: x (N), y (N), r (N).
    Objective: -sum(r)
    Constraints:
      1. Distance constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
      2. Boundary: xi >= ri, xi <= w-ri, yi >= ri, yi <= h-ri
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    def objective(v):
        if time.time() - start_time > max_time:
            return np.inf
        r = v[2*n:]
        return -np.sum(r)
    
    def jac(v):
        r = v[2*n:]
        return np.concatenate([np.zeros(n), np.zeros(n), -np.ones(n)])
    
    constraints = []
    
    # Pairwise distance constraints
    # To reduce constraint count, we can use a vectorized approach or just list them.
    # 21 circles -> 210 pairs. This is manageable.
    for i in range(n):
        for j in range(i + 1, n):
            def cons_dist(v, i=i, j=j):
                xi, yi, ri = v[i], v[n+i], v[2*n+i]
                xj, yj, rj = v[j], v[n+j], v[2*n+j]
                # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
            
            def jac_dist(v, i=i, j=j):
                xi, yi, ri = v[i], v[n+i], v[2*n+i]
                xj, yj, rj = v[j], v[n+j], v[2*n+j]
                jv = np.zeros(3*n)
                # d/dxi: 2(xi-xj), d/dyi: 2(yi-yj), d/dri: -2(ri+rj)
                jv[i] = 2*(xi-xj)
                jv[n+i] = 2*(yi-yj)
                jv[2*n+i] = -2*(ri+rj)
                
                jv[j] = -2*(xi-xj)
                jv[n+j] = -2*(yi-yj)
                jv[2*n+j] = -2*(ri+rj)
                return jv
            
            constraints.append({'type': 'ineq', 'fun': cons_dist, 'jac': jac_dist})
            
    # Boundary constraints
    # xi >= ri -> xi - ri >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i], 
                            'jac': lambda v, i=i: np.array([1.0 if k==i else (-1.0 if k==2*n+i else 0.0) for k in range(3*n)])})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (w - v[i]) - v[2*n+i], 
                            'jac': lambda v, i=i: np.array([-1.0 if k==i else (-1.0 if k==2*n+i else 0.0) for k in range(3*n)])})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i], 
                            'jac': lambda v, i=i: np.array([1.0 if k==n+i else (-1.0 if k==2*n+i else 0.0) for k in range(3*n)])})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (h - v[n+i]) - v[2*n+i], 
                            'jac': lambda v, i=i: np.array([-1.0 if k==n+i else (-1.0 if k==2*n+i else 0.0) for k in range(3*n)])})
        
    # Bounds for r
    bounds = []
    for i in range(n):
        bounds.append((0.0, w/2)) # x
        bounds.append((0.0, h/2)) # y
        bounds.append((0.0, min(w, h)/2)) # r
        
    v0 = np.concatenate([x0, y0, r0])
    
    try:
        res = minimize(objective, v0, jac=jac, method='trust-constr', 
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-8, 'verbose': 0})
        
        if res.success or res.fun < 0: # If it found a solution
            x_opt = res.x[:n]
            y_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            
            # Clip to ensure validity
            r_opt = np.maximum(r_opt, 1e-6)
            
            return x_opt, y_opt, r_opt
    except Exception as e:
        pass
    return None

def _pack(max_time=140.0):
    start_time = time.time()
    best_result = None
    best_value = -np.inf
    
    def evaluate(x, y, r):
        if not _check_valid(x, y, r):
            return None, -np.inf
        val = np.sum(r)
        return np.column_stack([x, y, r]), val
    
    # Candidate aspect ratios (w, h) with w+h=2
    # Perimeter 4 -> 2(w+h)=4 -> w+h=2
    aspects = [
        (1.0, 1.0),    # Square
        (1.2, 0.8),    # Wide
        (0.8, 1.2),    # Tall
        (1.3, 0.7),    # More wide
        (0.7, 1.3),    # More tall
    ]
    
    for w, h in aspects:
        if time.time() - start_time > max_time - 10:
            break
            
        # Initialize with a grid-like pattern scaled to w, h
        # Simple initialization: distribute points
        cols = int(np.sqrt(N * w/h))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        for i in range(N):
            row = i // cols
            col = i % cols
            x0[i] = (col + 0.5) * w / cols
            y0[i] = (row + 0.5) * h / rows
            r0[i] = min(w/cols, h/rows) * 0.45
            
        # Solve convex problem
        res = _solve_convex_pack(w, h, x0, y0, r0, max_time=max_time - 15, start_time=start_time)
        
        if res is not None:
            x_opt, y_opt, r_opt = res
            
            # Normalize to bounding box min 0,0
            x_min = np.min(x_opt - r_opt)
            y_min = np.min(y_opt - r_opt)
            x_opt -= x_min
            y_opt -= y_min
            
            result, val = evaluate(x_opt, y_opt, r_opt)
            if result is not None and val > best_value:
                best_value = val
                best_result = result
                
    # If convex solver failed or gave poor results, fallback to a simple grid
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                rr = 0.5 / side - 1e-4
                cells.append([cx, cy, rr])
        best_result = np.array(cells[:N])
        
    return best_result

def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    return _pack(max_time=140.0)

# EVOLVE-BLOCK-END
