import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Rectangle: w + h = 2 (perimeter = 4)
    # We'll try a few aspect ratios and pick the best
    
    best_result = None
    best_score = -1
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() - start_time > deadline:
            break
        
        # w + h = 2, w/h = ar => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        result = _pack_in_rectangle(w, h, N, deadline - (time.time() - start_time))
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def _pack_in_rectangle(w, h, n, time_budget):
    """Pack n circles in rectangle [0,w] x [0,h] maximizing sum of radii."""
    
    # Initial guess: hexagonal packing
    centers, radii = _hexagonal_init(n, w, h)
    
    if centers is None:
        return None
    
    # Variables: x1, y1, r1, x2, y2, r2, ..., xn, yn, rn
    x0 = np.concatenate([centers, radii])
    
    # Objective: maximize sum of radii => minimize -sum(radii)
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)
    
    # Constraints: non-overlap and boundary
    # For each pair (i,j): (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # Boundary: ri <= xi <= w-ri, ri <= yi <= h-ri
    
    def make_constraints(n):
        cons = []
        # Boundary constraints: xi >= ri, w - xi >= ri, yi >= ri, h - yi >= ri
        for i in range(n):
            # xi >= ri
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # w - xi >= ri
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[3*i] - v[3*i+2]})
            # yi >= ri
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # h - yi >= ri
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2]})
            # ri > 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
        
        # Non-overlap: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })
        
        return cons
    
    # Use penalty method instead for better performance
    def penalty_objective(vars, mu=100.0):
        radii = vars[2::3]
        base = -np.sum(radii)
        
        penalty = 0.0
        # Boundary violations
        for i in range(n):
            xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
            # Violations: xi < ri, w-xi < ri, yi < ri, h-yi < ri, ri < 0
            v1 = max(0, ri - xi)
            v2 = max(0, ri - (w - xi))
            v3 = max(0, ri - yi)
            v4 = max(0, ri - (h - yi))
            v5 = max(0, -ri)
            penalty += v1**2 + v2**2 + v3**2 + v4**2 + v5**2
        
        # Non-overlap violations
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                min_dist = ri + rj
                if dist_sq < min_dist**2:
                    viol = min_dist**2 - dist_sq
                    penalty += viol**2
        
        return base + mu * penalty
    
    # Try optimization with increasing penalty
    best_local = None
    best_local_score = -np.inf
    
    for mu in [1.0, 10.0, 100.0, 1000.0]:
        if time.time() - start_time > deadline:
            break
        
        try:
            res = minimize(penalty_objective, x0, method='BFGS', 
                          options={'maxiter': 2000, 'gtol': 1e-8},
                          args=(mu,))
            
            if res.success or res.fun < best_local_score:
                best_local = res.x
                best_local_score = res.fun
        
        except Exception:
            pass
    
    if best_local is None:
        return None
    
    # Extract result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i, 0] = best_local[3*i]
        result[i, 1] = best_local[3*i+1]
        result[i, 2] = max(0, best_local[3*i+2])
    
    # Verify validity
    if _is_valid(result, w, h):
        return result
    
    return None


def _hexagonal_init(n, w, h):
    """Generate hexagonal packing initial guess."""
    # Start with a grid and perturb slightly
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    if cols < 1 or rows < 1:
        return None
    
    dx = w / cols
    dy = h / rows
    
    centers = []
    radii = []
    
    for i in range(cols):
        for j in range(rows):
            if len(centers) >= n:
                break
            cx = (i + 0.5) * dx
            cy = (j + 0.5) * dy
            # Offset every other row for hexagonal
            if j % 2 == 1:
                cx += dx / 2
                cx = min(cx, w - dx/4)
                cx = max(cx, dx/4)
            
            r = min(dx, dy) / 2 * 0.95
            centers.append([cx, cy])
            radii.append(r)
    
    if len(centers) < n:
        # Fill remaining
        while len(centers) < n:
            centers.append([w/2, h/2])
            radii.append(min(w, h)/2 * 0.1)
    
    centers = np.array(centers[:n])
    radii = np.array(radii[:n])
    
    # Ensure circles are within bounds
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], w - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], h - radii[i])
    
    return centers, radii


def _is_valid(result, w, h, tol=1e-6):
    """Check if packing is valid."""
    n = len(result)
    
    for i in range(n):
        x, y, r = result[i]
        if r < -tol:
            return False
        if x < r - tol or x > w - r + tol:
            return False
        if y < r - tol or y > h - r + tol:
            return False
    
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = result[i]
            xj, yj, rj = result[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    
    return True
