import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 21

def _softplus(x, beta=100.0):
    """Smooth approximation of max(x, 0)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 1.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 1.0/beta, 1.0, 1.0 / (1.0 + np.exp(-beta * x)))

def _delaunay_edges(points):
    """Return pair indices from Delaunay triangulation."""
    try:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        if len(edges) == 0:
            # Fallback: all pairs
            idx = np.triu_indices(N, k=1)
            return idx[0], idx[1]
        ea = np.array([e[0] for e in edges])
        eb = np.array([e[1] for e in edges])
        return ea, eb
    except Exception:
        idx = np.triu_indices(N, k=1)
        return idx[0], idx[1]

def _hex_init():
    """Hexagonal packing initialization."""
    # For 21 circles, use a hexagonal arrangement
    # Rows: 4, 5, 4, 4, 4 (total 21) or similar
    # Let's use a more systematic approach
    rows = [4, 5, 4, 4, 4]
    total_rows = len(rows)
    max_cols = max(rows)
    
    # Calculate rectangle dimensions for hex packing
    # Horizontal spacing: 2r, vertical spacing: sqrt(3)*r
    # For perimeter 4: 2(w+h) = 4 => w + h = 2
    
    # Try to fit hex pattern
    r = 0.12  # initial radius guess
    h_spacing = 2 * r
    v_spacing = np.sqrt(3) * r
    
    xs = []
    ys = []
    for row_idx, n_cols in enumerate(rows):
        y = (row_idx + 0.5) * v_spacing
        offset = v_spacing / 2 if row_idx % 2 == 1 else 0
        for col in range(n_cols):
            x = (col + 0.5) * h_spacing + offset
            xs.append(x)
            ys.append(y)
    
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Center in rectangle
    w = 1.0
    h = 1.0
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-10) * (w - 2*r) + r
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-10) * (h - 2*r) + r
    
    return w, h, xs, ys, np.full(N, r)

def _grid_init():
    """Grid initialization."""
    side = 5
    w = 1.0
    h = 1.0
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xs[i] = (col + 0.5) / side * w
        ys[i] = (row + 0.5) / side * h
    r = w / side * 0.45
    return w, h, xs, ys, np.full(N, r)

def _random_init(seed):
    """Random initialization."""
    np.random.seed(seed)
    w = np.random.uniform(0.7, 1.3)
    h = 2.0 - w
    xs = np.random.uniform(0.05, w - 0.05, N)
    ys = np.random.uniform(0.05, h - 0.05, N)
    rs = np.random.uniform(0.08, 0.14, N)
    return w, h, xs, ys, rs

def _objective_and_grad(x, pair_i, pair_j):
    """Objective: minimize -sum(r) + penalties. Returns (val, grad)."""
    w = x[0]
    h = x[1]
    xs = x[2:2+N]
    ys = x[2+N:2+2*N]
    rs = x[2+2*N:2+3*N]
    
    sum_r = np.sum(rs)
    
    # Perimeter constraint: 2(w+h) <= 4 => w+h <= 2
    perim_viol = (w + h) - 2.0
    pen_perim = _softplus(perim_viol)
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    dx = xs[pair_i] - xs[pair_j]
    dy = ys[pair_i] - ys[pair_j]
    dist = np.sqrt(dx*dx + dy*dy + 1e-12)
    viol = rs[pair_i] + rs[pair_j] - dist
    pen_overlap = np.sum(_softplus(viol))
    
    # Boundary: r_i <= x_i, x_i + r_i <= w, r_i <= y_i, y_i + r_i <= h
    v_left = rs - xs
    v_right = xs + rs - w
    v_bottom = rs - ys
    v_top = ys + rs - h
    pen_bound = np.sum(_softplus(v_left)) + np.sum(_softplus(v_right)) + \
                np.sum(_softplus(v_bottom)) + np.sum(_softplus(v_top))
    
    val = -sum_r + 50.0 * (pen_perim + pen_overlap + pen_bound)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Perimeter
    g_perim = _softplus_grad(perim_viol)
    grad[0] += 50.0 * g_perim
    grad[1] += 50.0 * g_perim
    
    # Overlap
    g_viol = _softplus_grad(viol)
    coef = 50.0 * g_viol
    
    safe_dist = np.where(dist > 1e-10, dist, 1.0)
    gxi = -dx / safe_dist
    gxi = np.where(dist > 1e-10, gxi, 0.0)
    gyi = -dy / safe_dist
    gyi = np.where(dist > 1e-10, gyi, 0.0)
    
    np.add.at(grad, 2 + pair_i, coef * gxi)
    np.add.at(grad, 2 + pair_j, -coef * gxi)
    np.add.at(grad, 2 + N + pair_i, coef * gyi)
    np.add.at(grad, 2 + N + pair_j, -coef * gyi)
    np.add.at(grad, 2 + 2*N + pair_i, coef)
    np.add.at(grad, 2 + 2*N + pair_j, coef)
    
    # Boundary
    g_left = _softplus_grad(v_left)
    np.add.at(grad, 2 + np.arange(N), -50.0 * g_left)
    np.add.at(grad, 2 + 2*N + np.arange(N), 50.0 * g_left)
    
    g_right = _softplus_grad(v_right)
    np.add.at(grad, 2 + np.arange(N), 50.0 * g_right)
    np.add.at(grad, 2 + 2*N + np.arange(N), 50.0 * g_right)
    grad[0] -= 50.0 * np.sum(g_right)
    
    g_bottom = _softplus_grad(v_bottom)
    np.add.at(grad, 2 + N + np.arange(N), -50.0 * g_bottom)
    np.add.at(grad, 2 + 2*N + np.arange(N), 50.0 * g_bottom)
    
    g_top = _softplus_grad(v_top)
    np.add.at(grad, 2 + N + np.arange(N), 50.0 * g_top)
    np.add.at(grad, 2 + 2*N + np.arange(N), 50.0 * g_top)
    grad[1] -= 50.0 * np.sum(g_top)
    
    # Sum of radii
    grad[2+2*N:2+3*N] -= 1.0
    
    return val, grad

def _check_validity(x):
    """Check if solution is valid. Returns (is_valid, value)."""
    w, h = x[0], x[1]
    xs = x[2:2+N]
    ys = x[2+N:2+2*N]
    rs = x[2+2*N:2+3*N]
    
    if 2*(w+h) > 4 + 1e-7:
        return False, 0.0
    if w < 1e-6 or h < 1e-6:
        return False, 0.0
    
    for i in range(N):
        if xs[i] < -1e-7 or xs[i] > w + 1e-7:
            return False, 0.0
        if ys[i] < -1e-7 or ys[i] > h + 1e-7:
            return False, 0.0
        if rs[i] < 1e-7:
            return False, 0.0
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if d < rs[i] + rs[j] - 1e-7:
                return False, 0.0
    
    return True, np.sum(rs)

def _polish(x, pair_i, pair_j, time_remaining):
    """High-precision polish."""
    if time_remaining < 5:
        return x
    
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((0.001, 0.5))
    
    try:
        res = minimize(_objective_and_grad, x, jac=True, method='L-BFGS-B',
                      bounds=bounds, args=(pair_i, pair_j),
                      options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
        return res.x
    except Exception:
        return x

def construct():
    best_val = -np.inf
    best_x = None
    start_time = time.time()
    time_limit = 145  # Safety margin before 150s kill
    
    # Initial pair indices (will be updated as topology changes)
    pair_i, pair_j = np.triu_indices(N, k=1)
    
    # Multiple initialization strategies
    inits = []
    
    # Hexagonal
    w, h, xs, ys, rs = _hex_init()
    inits.append(np.concatenate(([w, h], xs, ys, rs)))
    
    # Grid
    w, h, xs, ys, rs = _grid_init()
    inits.append(np.concatenate(([w, h], xs, ys, rs)))
    
    # Random with different seeds
    for seed in [1, 7, 13, 42, 99, 123, 200, 333, 500, 777]:
        w, h, xs, ys, rs = _random_init(seed)
        inits.append(np.concatenate(([w, h], xs, ys, rs)))
    
    bounds = [(0.1, 1.9), (0.1, 1.9)]
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((0, 2))
    for _ in range(N):
        bounds.append((0.001, 0.5))
    
    for init_idx, x0 in enumerate(inits):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Update pair indices based on current positions
        points = np.column_stack((x0[2:2+N], x0[2+N:2+2*N]))
        pair_i, pair_j = _delaunay_edges(points)
        
        try:
            res = minimize(_objective_and_grad, x0, jac=True, method='L-BFGS-B',
                          bounds=bounds, args=(pair_i, pair_j),
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-11})
            x_opt = res.x
            
            valid, val = _check_validity(x_opt)
            if valid and val > best_val:
                best_val = val
                best_x = x_opt
        except Exception:
            pass
    
    # Polish best solution with updated topology
    if best_x is not None and time.time() - start_time < time_limit - 5:
        points = np.column_stack((best_x[2:2+N], best_x[2+N:2+2*N]))
        pair_i, pair_j = _delaunay_edges(points)
        
        time_remaining = time_limit - (time.time() - start_time)
        x_polished = _polish(best_x, pair_i, pair_j, time_remaining)
        
        valid, val_polished = _check_validity(x_polished)
        if valid and val_polished > best_val:
            best_val = val_polished
            best_x = x_polished
    
    if best_x is not None:
        w, h = best_x[0], best_x[1]
        xs = best_x[2:2+N]
        ys = best_x[2+N:2+2*N]
        rs = best_x[2+2*N:2+3*N]
        
        # Clamp to valid range
        rs = np.maximum(rs, 1e-6)
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        
        result = np.column_stack((xs, ys, rs))
        return result
    else:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append(((col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)
