import numpy as np
from scipy.optimize import linprog
import time

N = 32
TIME_LIMIT = 145.0  # Keep margin from 150s

def _elapsed(start):
    return time.time() - start

def _solve_radii_lp(x, y):
    """
    Solve the LP to maximize sum of radii given fixed centers x, y.
    Returns array of radii r.
    """
    # c: coefficients of objective function (minimize -sum(r))
    c = -np.ones(N)
    
    # A_ub: inequality constraints A_ub @ r <= b_ub
    # Constraints:
    # 1. r_i + r_j <= dist(i, j) for all i < j
    # 2. r_i <= bound_i for all i
    # 3. r_i >= 0 (handled by bounds)
    
    # Calculate distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    
    # Boundary constraints
    bounds_val = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    bounds_val = np.maximum(bounds_val, 0.0) # Ensure non-negative bounds for LP
    
    # Build constraint matrix
    # Number of pair constraints: N*(N-1)/2
    num_pairs = N * (N - 1) // 2
    A_ub = np.zeros((num_pairs + N, N))
    b_ub = np.zeros(num_pairs + N)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist[i, j]
            idx += 1
    
    # Boundary constraints: r_i <= bounds_val[i]
    for i in range(N):
        A_ub[idx, i] = 1.0
        b_ub[idx] = bounds_val[i]
        idx += 1
        
    # Bounds for r_i: 0 <= r_i <= bounds_val[i]
    res_bounds = [(0.0, bounds_val[i]) for i in range(N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=res_bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: return 0 or small values if LP fails (should not happen)
            return np.zeros(N)
    except Exception:
        return np.zeros(N)

def _is_valid(x, y, r, eps=1e-7):
    # bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # pairwise disjoint interiors
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy)
    min_dist = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist[mask] < min_dist[mask] - eps):
        return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _hex_init():
    # Hexagonal packing
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9: span = 1.0
    scale = 0.9 / span
    x = (pts[:,0] - xmin) * scale + 0.05
    y = (pts[:,1] - ymin) * scale + 0.05
    return x, y

def _grid_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    return x, y

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    return x, y

def _optimize(x, y, start):
    best_x, best_y, best_r = x.copy(), y.copy(), np.zeros(N)
    best_score = -1e10
    
    # Initial radii
    r = _solve_radii_lp(x, y)
    score = _sum_r(r)
    if score > best_score:
        best_score = score
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
    rng = np.random.RandomState(42)
    step_size = 0.05
    min_step = 0.001
    
    while _elapsed(start) < TIME_LIMIT - 5:
        # Perturb positions
        noise = rng.normal(0, step_size, (N, 2))
        new_x = x + noise[:, 0]
        new_y = y + noise[:, 1]
        
        # Clamp to [0, 1]
        new_x = np.clip(new_x, 0.0, 1.0)
        new_y = np.clip(new_y, 0.0, 1.0)
        
        # Solve for new radii
        new_r = _solve_radii_lp(new_x, new_y)
        new_score = _sum_r(new_r)
        
        # Accept if better (Hill Climbing)
        # To escape local minima, we could use Simulated Annealing, 
        # but Hill Climbing with many restarts/initializations is often sufficient 
        # and faster.
        if new_score > best_score:
            best_score = new_score
            best_x, best_y, best_r = new_x, new_y, new_r
            x, y, r = new_x, new_y, new_r
            # Reset step size occasionally to explore more
            if rng.random() < 0.05:
                step_size = max(min_step, step_size * 0.9)
        else:
            # Keep old x, y, r for next iteration
            # Optionally, we can do a "restart" from best if stuck for long time
            pass
            
        # Decay step size
        step_size = max(min_step, step_size * 0.999)
        
    return best_x, best_y, best_r

def construct():
    start = time.time()
    
    inits = []
    inits.append(_hex_init())
    inits.append(_grid_init(6, 6))
    inits.append(_grid_init(5, 7))
    inits.append(_grid_init(7, 5))
    inits.append(_grid_init(8, 4))
    inits.append(_grid_init(4, 8))
    inits.append(_random_init(1))
    inits.append(_random_init(2))
    inits.append(_random_init(3))
    
    best_x, best_y, best_r = inits[0][0], inits[0][1], np.zeros(N)
    best_score = -1e10
    
    # Initial evaluation for first init
    r = _solve_radii_lp(best_x, best_y)
    score = _sum_r(r)
    if score > best_score:
        best_score = score
        best_r = r
        
    for k, (x, y) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 10:
            break
            
        # Optimize from this initialization
        ox, oy, or_ = _optimize(x, y, start)
        
        # Verify validity of the result
        if _is_valid(ox, oy, or_):
            s = _sum_r(or_)
            if s > best_score:
                best_score = s
                best_x, best_y, best_r = ox, oy, or_
        else:
            # If invalid, try to repair or just skip
            # The LP solver should guarantee validity if bounds are respected.
            # But numerical errors might cause slight violations.
            # We can shrink radii slightly.
            or_ = or_ * 0.999
            if _is_valid(ox, oy, or_):
                s = _sum_r(or_)
                if s > best_score:
                    best_score = s
                    best_x, best_y, best_r = ox, oy, or_

    # Final polish: run optimization again from the best found so far
    if _elapsed(start) < TIME_LIMIT - 5:
        ox, oy, or_ = _optimize(best_x, best_y, start)
        if _is_valid(ox, oy, or_):
            s = _sum_r(or_)
            if s > best_score:
                best_x, best_y, best_r = ox, oy, or_
        else:
            or_ = or_ * 0.999
            if _is_valid(ox, oy, or_):
                s = _sum_r(or_)
                if s > best_score:
                    best_x, best_y, best_r = ox, oy, or_

    result = np.column_stack((best_x, best_y, best_r))
    return result
