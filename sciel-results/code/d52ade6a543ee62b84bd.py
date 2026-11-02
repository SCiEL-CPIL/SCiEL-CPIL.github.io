"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _circle_dist(c1, c2):
    """Distance between two circle centers."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def _objective_and_constraints(x, N):
    """
    x: array of length 2*N + 1
       [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN, w]
    where w is the width, and height is (4 - 2*w)/2 = 2 - w
    Perimeter constraint: 2*(w + h) = 4 => h = 2 - w
    So w in (0, 2), h in (0, 2)
    
    Objective: maximize sum of radii => minimize -sum(r)
    Constraints:
      - Circles don't overlap: dist(ci, cj) >= ri + rj
      - Circles inside rectangle: ri <= xi, ri <= w - xi, ri <= yi, ri <= h - yi
    """
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:3*N]
    w = x[3*N]
    h = 2.0 - w
    
    # Objective: negative sum of radii
    obj = -np.sum(radii)
    
    # Inequality constraints: g(x) >= 0
    # 1. Non-overlap: dist(ci, cj) - ri - rj >= 0
    # 2. Inside rectangle: xi - ri >= 0, w - xi - ri >= 0, yi - ri >= 0, h - yi - ri >= 0
    
    constraints = []
    
    # Non-overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            d = _circle_dist(centers[i], centers[j])
            constraints.append(d - radii[i] - radii[j])
    
    # Inside rectangle constraints
    for i in range(N):
        constraints.append(centers[i, 0] - radii[i])       # left
        constraints.append(w - centers[i, 0] - radii[i])   # right
        constraints.append(centers[i, 1] - radii[i])       # bottom
        constraints.append(h - centers[i, 1] - radii[i])   # top
    
    return obj, constraints

def _make_constraints_list(N):
    """Create scipy constraint objects."""
    cons = []
    
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            def non_overlap(x, i=i, j=j):
                ci = x[2*i:2*i+2]
                cj = x[2*j:2*j+2]
                ri = x[2*N+i]
                rj = x[2*N+j]
                d = np.sqrt(np.sum((ci - cj)**2))
                return d - ri - rj
            cons.append({'type': 'ineq', 'fun': non_overlap})
    
    # Inside rectangle
    for i in range(N):
        # Left: xi - ri >= 0
        def left(x, i=i):
            return x[2*i] - x[2*N+i]
        cons.append({'type': 'ineq', 'fun': left})
        
        # Right: w - xi - ri >= 0
        def right(x, i=i):
            return x[3*N] - x[2*i] - x[2*N+i]
        cons.append({'type': 'ineq', 'fun': right})
        
        # Bottom: yi - ri >= 0
        def bottom(x, i=i):
            return x[2*i+1] - x[2*N+i]
        cons.append({'type': 'ineq', 'fun': bottom})
        
        # Top: h - yi - ri >= 0, where h = 2 - w
        def top(x, i=i):
            h = 2.0 - x[3*N]
            return h - x[2*i+1] - x[2*N+i]
        cons.append({'type': 'ineq', 'fun': top})
    
    return cons

def _bounds(N):
    """Variable bounds: [x1, y1, r1, ..., xN, yN, rN, w]"""
    # x, y in [0, 2], r in [0, 1], w in [0.5, 1.5] (reasonable range)
    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))  # xi
        bounds.append((0.0, 2.0))  # yi
        bounds.append((0.0, 1.0))  # ri
    bounds.append((0.3, 1.7))  # w
    return bounds

def _grid_initialization(N):
    """
    Create initial grid-based configuration.
    Use a grid layout to get disjoint circles initially.
    """
    # Try different grid arrangements
    best_x = None
    best_obj = -np.inf
    
    for side in range(3, 8):
        if side * side < N:
            continue
        
        # Try different aspect ratios
        for w in np.linspace(0.5, 1.5, 11):
            h = 2.0 - w
            
            # Grid positions
            xs = np.linspace(0.5 * w / side, (side - 0.5) * w / side, side)
            ys = np.linspace(0.5 * h / side, (side - 0.5) * h / side, side)
            
            centers = []
            radii = []
            
            for i in range(side):
                for j in range(side):
                    if len(centers) >= N:
                        break
                    cx = xs[i]
                    cy = ys[j]
                    # Max radius limited by distance to walls and neighbors
                    r = min(cx, w - cx, cy, h - cy, w / (2 * side), h / (2 * side))
                    r = max(r, 0.01)
                    centers.append([cx, cy])
                    radii.append(r)
            
            if len(centers) < N:
                continue
            
            centers = np.array(centers[:N])
            radii = np.array(radii[:N])
            
            # Check if circles are disjoint (with small margin)
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    if d < radii[i] + radii[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            obj = -np.sum(radii)
            if obj < best_obj:
                best_obj = obj
                x = np.concatenate([centers.flatten(), radii, [w]])
                best_x = x
    
    if best_x is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        centers = []
        radii = []
        for i in range(side):
            for j in range(side):
                if len(centers) >= N:
                    break
                cx = (i + 0.5) * w / side
                cy = (j + 0.5) * h / side
                r = min(w, h) / (2 * side) - 1e-3
                centers.append([cx, cy])
                radii.append(max(r, 0.01))
        centers = np.array(centers[:N])
        radii = np.array(radii[:N])
        best_x = np.concatenate([centers.flatten(), radii, [w]])
    
    return best_x

def _evaluate(x, N):
    """Evaluate objective and check constraints."""
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:3*N]
    w = x[3*N]
    h = 2.0 - w
    
    obj = -np.sum(radii)
    
    # Check constraints
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < radii[i] + radii[j] - 1e-8:
                return obj, False
        if centers[i, 0] < radii[i] - 1e-8 or w - centers[i, 0] < radii[i] - 1e-8:
            return obj, False
        if centers[i, 1] < radii[i] - 1e-8 or h - centers[i, 1] < radii[i] - 1e-8:
            return obj, False
    
    return obj, True

def _polish(x, N, max_iter=2000):
    """High-precision polish pass."""
    bounds = _bounds(N)
    cons = _make_constraints_list(N)
    
    def obj_func(x):
        return -np.sum(x[2*N:3*N])
    
    try:
        res = minimize(obj_func, x, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': max_iter, 'ftol': 1e-12})
        if res.success:
            return res.x
        else:
            return x
    except Exception:
        return x

def construct():
    start_time = time.time()
    
    # Get initial configuration
    x0 = _grid_initialization(N)
    
    # Get bounds and constraints
    bounds = _bounds(N)
    cons = _make_constraints_list(N)
    
    def obj_func(x):
        return -np.sum(x[2*N:3*N])
    
    # First SLSQP optimization
    best_x = x0.copy()
    best_obj, best_valid = _evaluate(best_x, N)
    
    try:
        res = minimize(obj_func, best_x, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 1000, 'ftol': 1e-8})
        if res.x is not None:
            obj_val, valid = _evaluate(res.x, N)
            if valid and obj_val < best_obj:
                best_x = res.x.copy()
                best_obj = obj_val
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 30:
        return _format_result(best_x)
    
    # Polish pass
    polished = _polish(best_x, N, max_iter=3000)
    obj_val, valid = _evaluate(polished, N)
    if valid and obj_val < best_obj:
        best_x = polished
        best_obj = obj_val
    
    # Try a few more restarts if time permits
    n_restarts = 0
    while time.time() - start_time < DEADLINE - 20 and n_restarts < 5:
        # Perturb the best solution
        x_perturbed = best_x + np.random.normal(0, 0.01, size=best_x.shape)
        # Clip to bounds
        for i in range(len(bounds)):
            x_perturbed[i] = np.clip(x_perturbed[i], bounds[i][0], bounds[i][1])
        
        try:
            res = minimize(obj_func, x_perturbed, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 500, 'ftol': 1e-8})
            if res.x is not None:
                obj_val, valid = _evaluate(res.x, N)
                if valid and obj_val < best_obj:
                    best_x = res.x.copy()
                    best_obj = obj_val
        except Exception:
            pass
        n_restarts += 1
    
    # Final polish
    if time.time() - start_time < DEADLINE - 10:
        polished = _polish(best_x, N, max_iter=5000)
        obj_val, valid = _evaluate(polished, N)
        if valid and obj_val < best_obj:
            best_x = polished
            best_obj = obj_val
    
    return _format_result(best_x)

def _format_result(x):
    """Convert internal representation to (N, 3) array of (cx, cy, r)."""
    N = 21
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:3*N]
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    return result

# EVOLVE-BLOCK-END
