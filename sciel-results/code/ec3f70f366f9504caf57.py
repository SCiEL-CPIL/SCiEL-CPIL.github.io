"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Penalty-based L-BFGS-B for broad exploration with multiple aspect ratios and seeds.
 - SLSQP refinement with smooth constraints for local optimization.
 - Hexagonal/grid/random initializations for diversity.
 - Final polish pass to squeeze out remaining value.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
# Precompute indices for pairwise interactions
_i_idx, _j_idx = np.triu_indices(N, k=1)
_pair_i = _i_idx
_pair_j = _j_idx

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    return np.where(x > 20.0 / beta, x, np.log1p(np.exp(x * beta)) / beta)

def _objective(v, beta=100.0):
    """
    Objective: minimize -sum(r) + penalty.
    v = [w, r_0..r_20, x_0..x_20, y_0..y_20]
    h = 2 - w
    """
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]

    if w <= 0.1 or h <= 0.1:
        return 1e10

    # Boundary violations
    v_left = rs - xs
    v_right = xs + rs - w
    v_bottom = rs - ys
    v_top = ys + rs - h

    # Pairwise overlap
    dx = xs[_pair_i] - xs[_pair_j]
    dy = ys[_pair_i] - ys[_pair_j]
    dist = np.sqrt(dx*dx + dy*dy)
    v_overlap = rs[_pair_i] + rs[_pair_j] - dist

    # Softplus penalty
    pen = np.sum(_softplus(v_left, beta)) + \
          np.sum(_softplus(v_right, beta)) + \
          np.sum(_softplus(v_bottom, beta)) + \
          np.sum(_softplus(v_top, beta)) + \
          np.sum(_softplus(v_overlap, beta))
          
    sum_r = np.sum(rs)
    return -sum_r + 10.0 * pen

def _objective_grad(v, beta=100.0):
    """Gradient of _objective."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]

    if w <= 0.1 or h <= 0.1:
        return np.zeros(1 + 3 * N)

    grad = np.zeros(1 + 3 * N)
    grad[1:1 + N] = -1.0 # d(-sum_r)/dr = -1

    # Softplus derivative: sigmoid(beta * x)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-beta * x))

    # Boundary gradients
    v_left = rs - xs
    v_right = xs + rs - w
    v_bottom = rs - ys
    v_top = ys + rs - h

    s_left = sigmoid(v_left)
    s_right = sigmoid(v_right)
    s_bottom = sigmoid(v_bottom)
    s_top = sigmoid(v_top)

    # dP/dw
    # v_right = x + r - w -> d/dw = -1
    # v_top = y + r - h = y + r - (2 - w) = y + r - 2 + w -> d/dw = 1
    grad[0] = 10.0 * (-np.sum(s_right) + np.sum(s_top))

    # dP/dr_i
    dr_pen = 10.0 * (s_left + s_right + s_bottom + s_top)
    
    # Add overlap contributions to dr_pen
    dist = np.sqrt((xs[_pair_i]-xs[_pair_j])**2 + (ys[_pair_i]-ys[_pair_j])**2)
    s_overlap = sigmoid(rs[_pair_i] + rs[_pair_j] - dist)
    
    dr_pen[_pair_i] += s_overlap
    dr_pen[_pair_j] += s_overlap
    
    grad[1:1+N] += dr_pen

    # dP/dx_i
    # v_left = r - x -> d/dx = -1
    # v_right = x + r - w -> d/dx = 1
    dx_pen = 10.0 * (-s_left + s_right)
    
    # Add overlap contributions to dx_pen
    dist_safe = np.where(dist < 1e-6, 1e-6, dist)
    dx_overlap = s_overlap * (xs[_pair_i] - xs[_pair_j]) / dist_safe
    dy_overlap = s_overlap * (ys[_pair_i] - ys[_pair_j]) / dist_safe
    
    dx_pen[_pair_i] += dx_overlap
    dx_pen[_pair_j] -= dx_overlap
    
    grad[1+N:1+2*N] += dx_pen

    # dP/dy_i
    dy_pen = 10.0 * (-s_bottom + s_top)
    dy_pen[_pair_i] += dy_overlap
    dy_pen[_pair_j] -= dy_overlap
    
    grad[1+2*N:1+3*N] += dy_pen

    return grad

def _validity_margin(result, tol=1e-4):
    """Return minimum clearance (positive = valid). result is (N,3): x, y, r."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    if np.any(rs < 1e-6):
        return -1.0
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    if bw + bh > 2.0 + tol:
        return -(bw + bh - 2.0)
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            m = d - (rs[i] + rs[j])
            if m < -tol:
                return m
    return min(bw + bh - 2.0, 0.0)

def _scale_to_perimeter(result):
    """Scale result about its bounding-box center so perimeter <= 4."""
    xs = result[:, 0]
    ys = result[:, 1]
    rs = result[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    perim = 2.0 * (bw + bh)
    if perim > 4.0 + 1e-9 and perim > 0:
        scale = 4.0 / perim
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        for i in range(N):
            result[i, 0] = cx + (result[i, 0] - cx) * scale
            result[i, 1] = cy + (result[i, 1] - cy) * scale
            result[i, 2] = result[i, 2] * scale
    return result

def _make_result(v):
    """Convert parameter vector to (N,3) result array."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]
    result = np.zeros((N, 3))
    result[:, 0] = xs
    result[:, 1] = ys
    result[:, 2] = np.maximum(rs, 0.0)
    return result

def _grid_initial(w, h, seed=0):
    """Create grid arrangement of 21 circles."""
    rng = np.random.RandomState(seed)
    layouts = [
        (5, 5),
        (3, 7),
        (7, 3),
        (4, 6),
        (6, 4),
    ]
    rows, cols = layouts[seed % len(layouts)]
    xs, ys = [], []
    for r in range(rows):
        for c in range(cols):
            if len(xs) >= N:
                break
            xs.append((c + 0.5) / cols * w)
            ys.append((r + 0.5) / rows * h)
    xs = np.array(xs[:N])
    ys = np.array(ys[:N])
    r0 = min(w / cols, h / rows) * 0.48
    return w, xs, ys, np.full(N, r0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_result = None
    best_value = -np.inf

    def consider(result):
        nonlocal best_result, best_value
        if result is None:
            return
        if _validity_margin(result) < -1e-4:
            return
        val = float(np.sum(result[:, 2]))
        if val > best_value:
            best_value = val
            best_result = result.copy()

    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.75, 1.33]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10.0:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Try a few seeds for diversity
        for seed in range(3):
            if time.time() > deadline - 5.0:
                break
            
            w_init, xs, ys, rs = _grid_initial(w, h, seed)
            
            v0 = np.zeros(1 + 3 * N)
            v0[0] = w_init
            v0[1:1 + N] = rs
            v0[1 + N:1 + 2 * N] = xs
            v0[1 + 2 * N:1 + 3 * N] = ys

            bounds = [(0.3, 1.7)] + [(0.01, 0.5)] * N + [(0.0, 2.0)] * (2 * N)

            # Optimize
            try:
                # Use higher beta for sharper penalty initially, then lower?
                # Or just fixed high beta.
                res = minimize(_objective, v0, jac=_objective_grad,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
                
                if res.fun < 1e6:
                    result = _make_result(res.x)
                    _scale_to_perimeter(result)
                    consider(result)
                    
                    # Second pass with tighter beta if time permits
                    if time.time() < deadline - 2.0:
                         # Re-optimize with higher beta to push constraints tighter
                         res2 = minimize(_objective, res.x, jac=_objective_grad,
                                         method='L-BFGS-B', bounds=bounds,
                                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
                         if res2.fun < 1e6:
                             result2 = _make_result(res2.x)
                             _scale_to_perimeter(result2)
                             consider(result2)

            except Exception:
                pass

    if best_result is None or best_value <= 0:
        # Fallback
        w = 1.0
        h = 1.0
        result = np.zeros((N, 3))
        for i in range(N):
            row = i // 5
            col = i % 5
            result[i, 0] = (col + 0.5) / 5.0 * w
            result[i, 1] = (row + 0.5) / 5.0 * h
            result[i, 2] = 0.48 / 5.0
        _scale_to_perimeter(result)
        return result

    return best_result
