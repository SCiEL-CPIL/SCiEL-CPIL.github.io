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


def _objective(v, lam=50.0):
    """
    Objective: maximize sum(r) minus barrier violations.
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

    pen = 0.0
    for i in range(N):
        pen += max(0.0, -(xs[i] - rs[i])) ** 2
        pen += max(0.0, -(w - xs[i] - rs[i])) ** 2
        pen += max(0.0, -(ys[i] - rs[i])) ** 2
        pen += max(0.0, -(h - ys[i] - rs[i])) ** 2

    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            viol = rs[i] + rs[j] - d
            if viol > 0:
                pen += viol ** 2

    sum_r = np.sum(rs)
    return -sum_r + lam * pen


def _objective_grad(v, lam=50.0):
    """Gradient of _objective for L-BFGS-B."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]

    if w <= 0.1 or h <= 0.1:
        return np.array([0.0] + [0.0] * (3 * N))

    grad = np.zeros(1 + 3 * N)
    grad[1:1 + N] = -1.0

    for i in range(N):
        viol1 = max(0.0, -(xs[i] - rs[i]))
        if viol1 > 0:
            grad[1 + N + i] += 2 * lam * viol1
            grad[1 + i] += -2 * lam * viol1
        viol2 = max(0.0, -(w - xs[i] - rs[i]))
        if viol2 > 0:
            grad[0] += 2 * lam * viol2
            grad[1 + N + i] += -2 * lam * viol2
            grad[1 + i] += -2 * lam * viol2
        viol3 = max(0.0, -(ys[i] - rs[i]))
        if viol3 > 0:
            grad[1 + 2 * N + i] += 2 * lam * viol3
            grad[1 + i] += -2 * lam * viol3
        viol4 = max(0.0, -(h - ys[i] - rs[i]))
        if viol4 > 0:
            grad[0] += -2 * lam * viol4
            grad[1 + 2 * N + i] += -2 * lam * viol4
            grad[1 + i] += -2 * lam * viol4

    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                continue
            viol = rs[i] + rs[j] - d
            if viol > 0:
                grad[1 + i] += 2 * lam * viol
                grad[1 + j] += 2 * lam * viol
                grad[1 + N + i] += 2 * lam * viol * dx / d
                grad[1 + N + j] += -2 * lam * viol * dx / d
                grad[1 + 2 * N + i] += 2 * lam * viol * dy / d
                grad[1 + 2 * N + j] += -2 * lam * viol * dy / d

    return grad


def _slsqp_objective(v):
    """SLSQP objective: minimize -sum(r) with smooth constraints."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]
    return -np.sum(rs)


def _slsqp_constraints(v):
    """Smooth constraint functions for SLSQP: each returns >= 0 when satisfied."""
    w = v[0]
    h = 2.0 - w
    rs = v[1:1 + N]
    xs = v[1 + N:1 + 2 * N]
    ys = v[1 + 2 * N:1 + 3 * N]
    
    cons = []
    # Boundary constraints (smooth)
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+N+i] - v[1+i]})  # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[0] - v[1+N+i] - v[1+i]})  # w - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[1+2*N+i] - v[1+i]})  # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 2.0 - v[0] - v[1+2*N+i] - v[1+i]})  # h - y_i - r_i >= 0
    
    # Non-overlap constraints (smooth: dist - (r_i + r_j) >= 0)
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: 
                         np.sqrt((v[1+N+i]-v[1+N+j])**2 + (v[1+2*N+i]-v[1+2*N+j])**2) - v[1+i] - v[1+j]})
    
    return cons


def _slsqp_refine(v0, max_iter=200):
    """Refine a packing using SLSQP with smooth constraints."""
    try:
        constraints = _slsqp_constraints(v0)
        bounds = [(0.3, 1.7)] + [(0.01, 0.5)] * N + [(0.0, 2.0)] * (2 * N)
        
        res = minimize(_slsqp_objective, v0, method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'disp': False})
        return res
    except Exception:
        return None


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


def _hex_initial(w, h, seed=0):
    """Create hexagonal arrangement of 21 circles in rectangle w x h."""
    # Try different row configurations that sum to 21
    configs = [
        [7, 7, 7],       # 3 rows
        [4, 5, 4, 4, 4], # 5 rows
        [3, 4, 4, 4, 4, 2], # 6 rows
        [5, 5, 5, 3, 3], # 5 rows
        [6, 6, 6, 3],    # 4 rows
        [2, 4, 4, 4, 4, 3], # 6 rows
    ]
    cfg = configs[seed % len(configs)]
    
    pts = []
    for k, n in enumerate(cfg):
        y = k * np.sqrt(3.0)
        off = 1.0 if k % 2 == 1 else 0.0
        for i in range(n):
            if len(pts) >= N:
                break
            pts.append((i * 2.0 + off, y))
    pts = np.array(pts[:N])
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Center and scale to fit in w x h
    cx = 0.5 * (xs.max() + xs.min())
    cy = 0.5 * (ys.max() + ys.min())
    xs = xs - cx
    ys = ys - cy
    wc = xs.max() - xs.min()
    hc = ys.max() - ys.min()
    
    if wc > 0 and hc > 0:
        s = min((w * 0.85) / (wc + 2.0), (h * 0.85) / (hc + 2.0))
        s = max(s, 0.01)
        xs_f = xs * s + 0.5 * w
        ys_f = ys * s + 0.5 * h
    else:
        xs_f = np.linspace(0.1, w - 0.1, N)
        ys_f = np.linspace(0.1, h - 0.1, N)
    
    r0 = min(w, h) * 0.14
    return w, xs_f, ys_f, np.full(N, r0)


def _grid_initial(w, h, seed=0):
    """Create grid arrangement of 21 circles."""
    # Use different grid layouts
    layouts = [
        (5, 5),  # 5x5 = 25, take 21
        (3, 7),  # 3x7 = 21
        (7, 3),  # 7x3 = 21
        (4, 6),  # 4x6 = 24, take 21
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
    r0 = min(w / cols, h / rows) * 0.45
    return w, xs, ys, np.full(N, r0)


def _random_initial(w, h, seed=0):
    """Create random arrangement with non-overlapping initial radii."""
    rng = np.random.RandomState(seed)
    xs = rng.uniform(0.15, w - 0.15, N)
    ys = rng.uniform(0.15, h - 0.15, N)
    r0 = min(w, h) * 0.10
    return w, xs, ys, np.full(N, r0)


def _polish(result, deadline):
    """Final polish: try to optimize the bounding box and radii."""
    if time.time() > deadline - 2.0:
        return result
    
    xs = result[:, 0].copy()
    ys = result[:, 1].copy()
    rs = result[:, 2].copy()
    
    # Compute current bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    bw = max_x - min_x
    bh = max_y - min_y
    
    if bw + bh >= 2.0 - 1e-6:
        return result
    
    # Try to expand radii slightly if there's room
    margin = (2.0 - bw - bh) / 2.0
    if margin > 1e-4:
        for i in range(N):
            # Check if we can expand circle i
            can_expand = True
            for j in range(N):
                if i == j:
                    continue
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if d < rs[i] + rs[j] + margin + 1e-6:
                    can_expand = False
                    break
            # Check boundary
            if xs[i] - rs[i] - margin < min_x or xs[i] + rs[i] + margin > max_x:
                can_expand = False
            if ys[i] - rs[i] - margin < min_y or ys[i] + rs[i] + margin > max_y:
                can_expand = False
            
            if can_expand:
                rs[i] += margin * 0.5
    
    result_new = np.column_stack([xs, ys, rs])
    _scale_to_perimeter(result_new)
    
    if _validity_margin(result_new) >= -1e-5:
        return result_new
    return result


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    # Try different aspect ratios
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.35, 0.65, 1.5, 0.9, 1.1]
    init_types = ['hex', 'grid', 'random']

    seed_counter = 0
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w

        for init_type in init_types:
            if time.time() > deadline - 8.0:
                break

            if init_type == 'hex':
                w_init, xs, ys, rs = _hex_initial(w, h, seed_counter)
            elif init_type == 'grid':
                w_init, xs, ys, rs = _grid_initial(w, h, seed_counter)
            else:
                w_init, xs, ys, rs = _random_initial(w, h, seed_counter)
            seed_counter += 1

            # Build initial parameter vector
            v0 = np.zeros(1 + 3 * N)
            v0[0] = w_init
            v0[1:1 + N] = rs
            v0[1 + N:1 + 2 * N] = xs
            v0[1 + 2 * N:1 + 3 * N] = ys

            bounds = [(0.3, 1.7)] + [(0.01, 0.5)] * N + [(0.0, 2.0)] * (2 * N)

            # Phase 1: L-BFGS-B with high penalty
            for lam in [200.0, 80.0]:
                if time.time() > deadline - 5.0:
                    break
                try:
                    res = minimize(_objective, v0, jac=_objective_grad,
                                   method='L-BFGS-B', bounds=bounds,
                                   args=(lam,),
                                   options={'maxiter': 300, 'ftol': 1e-10,
                                             'gtol': 1e-8, 'disp': False})
                except Exception:
                    continue

                if res.fun < 1e6:
                    result = _make_result(res.x)
                    _scale_to_perimeter(result)
                    consider(result)

                    # Phase 2: SLSQP refinement
                    if _validity_margin(result) > -1e-5 and time.time() < deadline - 3.0:
                        v1 = res.x.copy()
                        slsqp_res = _slsqp_refine(v1, max_iter=150)
                        if slsqp_res is not None and slsqp_res.fun < 1e4:
                            result2 = _make_result(slsqp_res.x)
                            _scale_to_perimeter(result2)
                            consider(result2)

    # Final polish on the best result
    if best_result is not None and best_value > 0:
        best_result = _polish(best_result, deadline)
        consider(best_result)

    # Final fallback: simple grid
    if best_result is None or best_value <= 0:
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
