"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 150.0
SAFETY_MARGIN = 15.0


def _compute_score(circles):
    """Compute the sum of radii for a valid packing, or 0 if invalid."""
    if circles is None or len(circles) == 0:
        return 0.0
    n = len(circles)
    if n != N:
        return 0.0
    
    # Check for NaN/inf
    if not np.all(np.isfinite(circles)):
        return 0.0
    
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Radii must be positive
    if np.any(r <= 0):
        return 0.0
    
    # Bounding rectangle
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        return 0.0
    
    perimeter = 2 * (width + height)
    if perimeter > 4.0 + 1e-9:
        return 0.0
    
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0
    
    return np.sum(r)


def _objective_and_constraints(x, aspect):
    """
    x: [x0, y0, r0, x1, y1, r1, ...]
    aspect: width/height ratio
    
    We optimize to maximize sum(r) subject to:
    - All circles inside rectangle of perimeter 4
    - Non-overlap
    """
    n = N
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Rectangle dimensions
    # Perimeter = 2*(W + H) = 4, so W + H = 2
    # aspect = W/H, so W = aspect*H
    # aspect*H + H = 2 => H = 2/(1+aspect), W = 2*aspect/(1+aspect)
    H = 2.0 / (1.0 + aspect)
    W = 2.0 * aspect / (1.0 + aspect)
    
    # Objective: maximize sum of radii (minimize negative)
    obj = -np.sum(rs)
    
    # Constraints:
    # 1. Boundary: xi >= ri, xi <= W - ri, yi >= ri, yi <= H - ri
    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    # For scipy minimize with SLSQP, we use inequality constraints g(x) >= 0
    cons = []
    
    # Boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[3*i+1] - x[3*i+2]})
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2 - (x[3*i+2]+x[3*j+2])**2
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W/2))  # xi
        bounds.append((1e-6, H/2))  # yi
        bounds.append((1e-6, min(W,H)/2))  # ri
    
    return obj, cons, bounds


def _initial_guess(aspect, n_circles):
    """Generate initial guess for circle positions and radii."""
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    
    # Grid-based initial guess
    cols = int(np.ceil(np.sqrt(n_circles * W / H)))
    rows = int(np.ceil(n_circles / cols))
    
    x = np.zeros(3 * n_circles)
    
    for k in range(n_circles):
        i = k % cols
        j = k // cols
        cx = (i + 0.5) * W / cols
        cy = (j + 0.5) * H / rows
        r = 0.4 * min(W / cols, H / rows)
        
        x[3*k] = cx
        x[3*k+1] = cy
        x[3*k+2] = r
    
    return x


def _solve_for_aspect(aspect, x0=None, time_limit=10.0):
    """Solve the optimization problem for a given aspect ratio."""
    start_time = time.time()
    
    if x0 is None:
        x0 = _initial_guess(aspect, N)
    
    try:
        obj, cons, bounds = _objective_and_constraints(x0, aspect)
        
        result = minimize(
            lambda x: _objective_and_constraints(x, aspect)[0],
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.success:
            circles = np.column_stack([result.x[0::3], result.x[1::3], result.x[2::3]])
            score = _compute_score(circles)
            if score > 0:
                return circles, score
        
        # Try with different initializations
        for attempt in range(3):
            if time.time() - start_time > time_limit:
                break
            x0_perturbed = x0 + np.random.normal(0, 0.01, size=x0.shape)
            x0_perturbed[2::3] = np.maximum(x0_perturbed[2::3], 1e-6)
            
            try:
                result = minimize(
                    lambda x: _objective_and_constraints(x, aspect)[0],
                    x0_perturbed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if result.success:
                    circles = np.column_stack([result.x[0::3], result.x[1::3], result.x[2::3]])
                    score = _compute_score(circles)
                    if score > 0:
                        return circles, score
            except:
                pass
        
    except Exception:
        pass
    
    return None, 0.0


def _grid_packing(aspect):
    """Simple grid packing as fallback."""
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    
    # Find grid dimensions
    best = None
    for cols in range(1, N + 1):
        rows = int(np.ceil(N / cols))
        cell_w = W / cols
        cell_h = H / rows
        r = 0.49 * min(cell_w, cell_h)
        
        circles = []
        for k in range(N):
            i = k % cols
            j = k // cols
            circles.append([(i + 0.5) * cell_w, (j + 0.5) * cell_h, r])
        
        score = _compute_score(np.array(circles))
        if best is None or score > best[1]:
            best = (np.array(circles), score)
    
    return best


def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_circles = None
    best_score = 0.0
    
    # Try a range of aspect ratios
    aspects = np.logspace(-1, 1, 15)  # From 0.1 to 10
    
    # Sort by how close to square (square often optimal)
    aspects_sorted = sorted(aspects, key=lambda a: abs(a - 1.0))
    
    for aspect in aspects_sorted:
        if time.time() > deadline:
            break
        
        # First try grid packing as baseline
        grid_result = _grid_packing(aspect)
        if grid_result is not None and grid_result[1] > best_score:
            best_circles = grid_result[0]
            best_score = grid_result[1]
        
        # Then try optimization
        circles, score = _solve_for_aspect(aspect, time_limit=min(8.0, max(2.0, deadline - time.time())))
        if circles is not None and score > best_score:
            best_circles = circles
            best_score = score
        
        if time.time() > deadline:
            break
    
    # If no valid packing found, use a simple square grid
    if best_circles is None or best_score <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        best_score = _compute_score(best_circles)
    
    # Scale to fit perimeter exactly if needed
    if best_circles is not None and best_score > 0:
        x = best_circles[:, 0]
        y = best_circles[:, 1]
        r = best_circles[:, 2]
        
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        
        W = x_max - x_min
        H = y_max - y_min
        P = 2 * (W + H)
        
        if P > 4.0:
            scale = 4.0 / P
            x = x * scale
            y = y * scale
            r = r * scale
            best_circles = np.column_stack([x, y, r])
    
    return best_circles
