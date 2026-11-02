"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, tol=1e-7):
    """Check if circles are valid: disjoint, inside bounding rect with perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(rs < -1e-9):
        return False
    rs = np.maximum(rs, 0)
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + tol:
        return False
    # Check pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _objective_soft(circles, W, H, beta=80.0):
    """
    Smooth objective: minimize -sum(r) + soft penalties.
    Uses exp barriers for separation and boundary violations.
    """
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    n = len(rs)
    
    # Boundary violations (smooth)
    # Circle i must satisfy: x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H
    viol = np.maximum(0, -xs + rs) + np.maximum(0, xs + rs - W) + \
           np.maximum(0, -ys + rs) + np.maximum(0, ys + rs - H)
    # Use soft version: sum of exp
    boundary_pen = 0.0
    for i in range(n):
        v = np.maximum(0, -xs[i] + rs[i])
        boundary_pen += v * v
        v = np.maximum(0, xs[i] + rs[i] - W)
        boundary_pen += v * v
        v = np.maximum(0, -ys[i] + rs[i])
        boundary_pen += v * v
        v = np.maximum(0, ys[i] + rs[i] - H)
        boundary_pen += v * v
    
    # Separation: for each pair, dist >= r_i + r_j
    sep_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            req = rs[i] + rs[j]
            viol = max(0, req - d)
            sep_pen += viol * viol
    
    return -np.sum(rs) + 1e4 * (boundary_pen + sep_pen)

def _objective_hard(circles, W, H, lam=1e5):
    """Hard penalty objective for final polish."""
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    n = len(rs)
    
    boundary_pen = 0.0
    for i in range(n):
        boundary_pen += max(0, -xs[i] + rs[i])**2
        boundary_pen += max(0, xs[i] + rs[i] - W)**2
        boundary_pen += max(0, -ys[i] + rs[i])**2
        boundary_pen += max(0, ys[i] + rs[i] - H)**2
    
    sep_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            viol = max(0, rs[i] + rs[j] - d)
            sep_pen += viol * viol
    
    return -np.sum(rs) + lam * (boundary_pen + sep_pen)

def _make_hex_grid(W, H, n_rows, n_cols):
    """Create a hexagonal grid of circles in rectangle [0,W]x[0,H]."""
    circles = np.zeros((N, 3))
    # Distribute N circles in n_rows x n_cols grid
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= N:
                break
            # Hex offset
            x_off = 0.0
            if row % 2 == 1:
                x_off = 0.5 * (W / n_cols)
            cx = (col + 0.5 + x_off) * W / n_cols
            cy = (row + 0.5) * H / n_rows
            # Radius: limited by cell size
            r = 0.5 * min(W / n_cols, H / n_rows) * 0.98
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r
            idx += 1
    # Fill remaining with small circles at center
    while idx < N:
        circles[idx, 0] = W / 2
        circles[idx, 1] = H / 2
        circles[idx, 2] = 0.01
        idx += 1
    return circles

def _sa_optimize(circles, W, H, max_iter=8000, T0=0.05, T_end=1e-6, deadline=None):
    """Simulated annealing to optimize circle packing."""
    best = circles.copy()
    best_val = -np.sum(best[:, 2])
    current = circles.copy()
    current_val = _objective_soft(current, W, H, beta=50)
    
    rng = np.random.RandomState(12345)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        T = T0 * (T_end / T0) ** (it / max_iter)
        
        # Propose a move
        move_type = rng.random()
        if move_type < 0.4:
            # Move a circle
            i = rng.randint(N)
            delta = T * 0.3 * rng.randn(2)
            new_c = current.copy()
            new_c[i, 0] += delta[0]
            new_c[i, 1] += delta[1]
        elif move_type < 0.7:
            # Change a radius
            i = rng.randint(N)
            delta_r = T * 0.2 * rng.randn()
            new_c = current.copy()
            new_c[i, 2] = max(0.001, new_c[i, 2] + delta_r)
        else:
            # Move two circles
            i, j = rng.choice(N, size=2, replace=False)
            delta_i = T * 0.3 * rng.randn(2)
            delta_j = T * 0.3 * rng.randn(2)
            new_c = current.copy()
            new_c[i, 0] += delta_i[0]
            new_c[i, 1] += delta_i[1]
            new_c[j, 0] += delta_j[0]
            new_c[j, 1] += delta_j[1]
        
        new_val = _objective_soft(new_c, W, H, beta=50)
        delta_val = new_val - current_val
        
        if delta_val < 0 or rng.random() < np.exp(-delta_val / T):
            current = new_c
            current_val = new_val
            if -np.sum(current[:, 2]) < best_val:
                best_val = -np.sum(current[:, 2])
                best = current.copy()
    
    return best, best_val

def _lbfgs_polish(circles, W, H, deadline=None):
    """Final L-BFGS polish."""
    params0 = circles.flatten()
    
    def obj(p):
        c = p.reshape(N, 3)
        return _objective_hard(c, W, H, lam=1e5)
    
    try:
        res = minimize(obj, params0, method='L-BFGS-B',
                       options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14})
        new_c = res.x.reshape(N, 3)
        if _check_valid(new_c):
            return new_c
    except Exception:
        pass
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    best_circles = None
    best_sum = 0.0
    
    # Try different aspect ratios and grid configurations
    configs = [
        # (W, H, n_rows, n_cols)
        (1.25, 0.75, 4, 5),    # 5:4 ratio, 4x5 grid
        (0.75, 1.25, 5, 4),    # 4:5 ratio, 5x4 grid
        (1.0, 1.0, 5, 5),      # square 5x5
        (1.333, 0.667, 3, 7),  # 2:1 ratio
        (0.667, 1.333, 7, 3),  # 1:2 ratio
        (1.2, 0.8, 4, 5),
        (0.8, 1.2, 5, 4),
        (1.1, 0.9, 4, 5),
        (0.9, 1.1, 5, 4),
    ]
    
    for W, H, n_rows, n_cols in configs:
        if time.time() > deadline - 10:
            break
        
        # Create initial hex grid
        circles0 = _make_hex_grid(W, H, n_rows, n_cols)
        
        # Run SA
        sa_circles, sa_val = _sa_optimize(circles0, W, H, max_iter=5000, deadline=deadline - 10)
        
        # Check if valid (inside rectangle)
        # Scale to fit: the bounding box should be within [0,W]x[0,H]
        # After SA, circles might be slightly outside; project back
        xs, ys, rs = sa_circles[:, 0], sa_circles[:, 1], sa_circles[:, 2]
        # Project centers to keep circles inside
        xs = np.clip(xs, rs, W - rs)
        ys = np.clip(ys, rs, H - rs)
        sa_circles[:, 0] = xs
        sa_circles[:, 1] = ys
        
        # Verify validity
        if _check_valid(sa_circles):
            s = np.sum(sa_circles[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = sa_circles.copy()
        
        # L-BFGS polish
        if time.time() < deadline - 5:
            polished = _lbfgs_polish(sa_circles, W, H, deadline=deadline - 2)
            if _check_valid(polished):
                s = np.sum(polished[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = polished.copy()
    
    # Final high-precision polish on best solution
    if best_circles is not None and time.time() < deadline - 3:
        # Determine the rectangle from the best solution
        xs, ys, rs = best_circles[:, 0], best_circles[:, 1], best_circles[:, 2]
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        W = max_x - min_x
        H = max_y - min_y
        # Shift to origin
        best_circles[:, 0] -= min_x
        best_circles[:, 1] -= min_y
        
        polished = _lbfgs_polish(best_circles, W, H, deadline=deadline - 1)
        if _check_valid(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = polished.copy()
    
    if best_circles is not None:
        best_circles[:, 2] = np.maximum(best_circles[:, 2], 0.0)
        return best_circles
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
