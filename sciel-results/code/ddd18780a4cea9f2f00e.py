"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _initial_grid(n, w, h):
    """Create a grid of n circles in a w x h rectangle with disjoint interiors."""
    if n <= 0:
        return np.zeros((0, 3))
    # Choose grid dimensions
    best = None
    for rows in range(1, n + 1):
        cols = (n + rows - 1) // rows
        area = rows * cols
        if area < n:
            continue
        # Score: prefer aspect ratio close to w/h
        aspect_grid = cols / rows
        aspect_rect = w / h
        score = abs(aspect_grid - aspect_rect) + 0.1 * (area - n) / n
        if best is None or score < best[0]:
            best = (score, rows, cols)
    if best is None:
        rows, cols = 1, n
    else:
        rows, cols = best[1], best[2]
    
    cells = []
    for i in range(rows):
        for j in range(cols):
            if len(cells) >= n:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            r = min(w / (2 * cols), h / (2 * rows)) * 0.95  # slightly smaller to ensure disjoint
            cells.append([x, y, r])
    return np.array(cells[:n])


def _compute_objective(params, n):
    """Compute the objective: sum of radii, with penalty for constraint violations."""
    x, y, r, w = params[:n], params[n:2*n], params[2*n:3*n], params[3*n]
    h = 2.0 - w  # perimeter constraint: 2(w+h) = 4 => h = 2 - w
    
    if w <= 0 or h <= 0:
        return 1e10
    
    penalty = 0.0
    
    # Boundary constraints
    for i in range(n):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > w:
            penalty += (x[i] + r[i] - w) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > h:
            penalty += (y[i] + r[i] - h) ** 2
    
    # Disjointness constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    obj = -np.sum(r) + 1e6 * penalty
    return obj


def _construct_single(w_init, h_init, max_iter=500, timeout=30):
    """Run a single optimization starting from grid with given aspect ratio."""
    n = N
    cells = _initial_grid(n, w_init, h_init)
    x0 = cells[:, 0].copy()
    y0 = cells[:, 1].copy()
    r0 = cells[:, 2].copy()
    
    # Ensure r0 is positive
    r0 = np.maximum(r0, 1e-6)
    
    x0 = np.clip(x0, r0, w_init - r0)
    y0 = np.clip(y0, r0, h_init - r0)
    
    params0 = np.concatenate([x0, y0, r0, [w_init]])
    
    # Bounds
    r_min = 1e-4
    bounds = []
    for i in range(n):
        bounds.append((0, w_init))  # x
    for i in range(n):
        bounds.append((0, h_init))  # y
    for i in range(n):
        bounds.append((r_min, min(w_init, h_init) / 2))  # r
    bounds.append((0.1, 1.9))  # w
    
    # Constraints
    constraints = []
    # Non-negativity of radii (already in bounds)
    
    start_time = time.time()
    
    try:
        result = minimize(
            _compute_objective,
            params0,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
        )
    except Exception:
        return None
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        return None
    
    if not result.success and result.fun > 1e8:
        return None
    
    params = result.x
    x, y, r, w = params[:n], params[n:2*n], params[2*n:3*n], params[3*n]
    h = 2.0 - w
    
    # Validate
    if w <= 0 or h <= 0:
        return None
    if np.any(r < 1e-5):
        return None
    
    # Check constraints
    valid = True
    for i in range(n):
        if x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
            valid = False
            break
        if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Try to repair by shrinking radii
        r = np.maximum(r * 0.9, 1e-5)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
    
    return np.column_stack([x, y, r]), np.sum(r), w, h


def construct():
    start_time = time.time()
    deadline = 140  # safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Try different initial aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # w/h = ar => w = ar*h, 2(w+h)=4 => h(2+2ar)=4 => h=2/(1+ar), w=2ar/(1+ar)
        w_init = 2.0 * ar / (1.0 + ar)
        h_init = 2.0 / (1.0 + ar)
        
        result = _construct_single(w_init, h_init, max_iter=300, timeout=15)
        if result is not None:
            circles, score, w, h = result
            if score > best_score:
                best_score = score
                best_result = circles
    
    # If we have time, try a few more with different seeds
    if time.time() < deadline - 20:
        np.random.seed(42)
        for trial in range(3):
            if time.time() > deadline:
                break
            ar = 0.5 + np.random.rand() * 1.5
            w_init = 2.0 * ar / (1.0 + ar)
            h_init = 2.0 / (1.0 + ar)
            result = _construct_single(w_init, h_init, max_iter=200, timeout=15)
            if result is not None:
                circles, score, w, h = result
                if score > best_score:
                    best_score = score
                    best_result = circles
    
    if best_result is None:
        # Fallback: simple grid
        cells = _initial_grid(N, 1.0, 1.0)
        return cells
    
    return best_result

# EVOLVE-BLOCK-END
