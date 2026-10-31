"""Circle packing in perimeter-4 rectangle: SLSQP refinement + penalty gradient + SA."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0  # safety margin before 150s kill

def _evaluate(x, W, H):
    """Return sum of radii if x is a valid packing, else None."""
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-7 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _slsqp_optimize(W, H, x0, maxiter=200):
    """
    SLSQP optimization on the (r, cx, cy) vector.
    x0: (N, 3) array [cx, cy, r]
    Returns (best_x, best_sum) or (None, 0.0) on failure.
    """
    n = N
    # Convert to SLSQP format: [r_0..r_{n-1}, cx_0..cx_{n-1}, cy_0..cy_{n-1}]
    x = np.zeros(3 * n)
    x[:n] = x0[:, 2]
    x[n:2*n] = x0[:, 0]
    x[2*n:3*n] = x0[:, 1]
    
    def objective(x):
        return -np.sum(x[:n])
    
    def constraints():
        cons = []
        # r_i >= 1e-4
        for i in range(n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - 1e-4,
            })
        # Boundary constraints
        for i in range(n):
            # cx_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[n + i] - x[i],
            })
            # (W - cx_i) - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (W - x[n + i]) - x[i],
            })
            # cy_i - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[2*n + i] - x[i],
            })
            # (H - cy_i) - r_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: (H - x[2*n + i]) - x[i],
            })
        # Pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (
                        (x[n + i] - x[n + j]) ** 2 +
                        (x[2*n + i] - x[2*n + j]) ** 2 -
                        (x[i] + x[j]) ** 2
                    ),
                })
        return cons
    
    cons = constraints()
    bounds = []
    for i in range(n):
        bounds.append((1e-4, min(W, H) / 2.0))
    for i in range(n):
        bounds.append((0.0, W))
    for i in range(n):
        bounds.append((0.0, H))
    
    try:
        res = minimize(
            objective,
            x,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-10},
        )
    except Exception:
        return None, 0.0
    
    x_opt = res.x
    r = np.maximum(x_opt[:n], 1e-4)
    cx = np.clip(x_opt[n:2*n], 0.0, W)
    cy = np.clip(x_opt[2*n:3*n], 0.0, H)
    
    out = np.column_stack([cx, cy, r])
    s = _evaluate(out, W, H)
    if s is not None:
        return out, s
    return None, 0.0

def _penalty_and_gradient(x, W, H, penalty_weight=100.0):
    """
    Compute smooth penalty objective and its analytic gradient.
    """
    cx = x[:, 0].copy()
    cy = x[:, 1].copy()
    r = x[:, 2].copy()
    n = len(cx)
    
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    penalty = 0.0
    grad = np.zeros_like(x)
    grad[:, 2] -= 1.0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy + 1e-12)
            req = r[i] + r[j]
            if d < req:
                overlap = req - d
                penalty += overlap * overlap
                if d > 1e-10:
                    factor = -2.0 * overlap
                    grad[i, 0] += penalty_weight * factor * (dx / d)
                    grad[i, 1] += penalty_weight * factor * (dy / d)
                    grad[j, 0] += penalty_weight * factor * (-dx / d)
                    grad[j, 1] += penalty_weight * factor * (-dy / d)
                    grad[i, 2] += penalty_weight * (-2.0 * overlap)
                    grad[j, 2] += penalty_weight * (-2.0 * overlap)
    
    for i in range(n):
        d_left = cx[i]
        d_right = W - cx[i]
        d_bottom = cy[i]
        d_top = H - cy[i]
        min_d = min(d_left, d_right, d_bottom, d_top)
        if r[i] > min_d:
            viol = r[i] - min_d
            penalty += viol * viol
            grad[i, 2] += penalty_weight * 2.0 * viol
            if min_d == d_left:
                grad[i, 0] -= penalty_weight * 2.0 * viol
            elif min_d == d_right:
                grad[i, 0] += penalty_weight * 2.0 * viol
            elif min_d == d_bottom:
                grad[i, 1] -= penalty_weight * 2.0 * viol
            elif min_d == d_top:
                grad[i, 1] += penalty_weight * 2.0 * viol
    
    return -float(np.sum(r)) + penalty_weight * penalty, grad

def _gradient_descent(x0, W, H, max_iter=300, lr=0.01, penalty_weight=50.0):
    """Gradient descent on the penalty objective."""
    n = len(x0)
    x = x0.copy()
    best_x = x.copy()
    best_sum = _evaluate(x, W, H) or 0.0
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        
        obj, grad = _penalty_and_gradient(x, W, H, penalty_weight)
        x_new = x - lr * grad
        
        x_new[:, 0] = np.clip(x_new[:, 0], 0.0, W)
        x_new[:, 1] = np.clip(x_new[:, 1], 0.0, H)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        lo = np.minimum(np.minimum(x_new[:, 0], x_new[:, 1]), np.minimum(W - x_new[:, 0], H - x_new[:, 1]))
        x_new[:, 2] = np.minimum(x_new[:, 2], lo)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        
        s_new = _evaluate(x_new, W, H)
        if s_new is not None and s_new > best_sum:
            best_sum = s_new
            best_x = x_new.copy()
        
        obj_new, _ = _penalty_and_gradient(x_new, W, H, penalty_weight)
        if obj_new < obj:
            x = x_new
            lr *= 1.01
        else:
            lr *= 0.9
        
        lr = max(lr, 1e-5)
        lr = min(lr, 0.05)
    
    return best_x, best_sum

def _sa_exploration(x0, W, H, steps=1000, T0=0.05, T1=0.001):
    """Simulated annealing for global exploration."""
    n = len(x0)
    cx = x0[:, 0].copy()
    cy = x0[:, 1].copy()
    r = x0[:, 2].copy()
    
    best_cx = cx.copy()
    best_cy = cy.copy()
    best_r = r.copy()
    best_sum = _evaluate(np.column_stack([best_cx, best_cy, best_r]), W, H) or 0.0
    
    current_sum = best_sum
    
    for step in range(steps):
        if time.time() > DEADLINE - 3:
            break
        T = T0 * (T1 / T0) ** (step / steps)
        
        cx_new = cx.copy()
        cy_new = cy.copy()
        r_new = r.copy()
        
        n_move = random.randint(1, 2)
        for _ in range(n_move):
            i = random.randint(0, n - 1)
            cx_new[i] += random.gauss(0, 0.03)
            cy_new[i] += random.gauss(0, 0.03)
            if random.random() < 0.4:
                r_new[i] = max(1e-6, r_new[i] * random.uniform(0.85, 1.15))
        
        cx_new = np.clip(cx_new, 0.0, W)
        cy_new = np.clip(cy_new, 0.0, H)
        r_new = np.maximum(r_new, 1e-6)
        lo = np.minimum(np.minimum(cx_new, cy_new), np.minimum(W - cx_new, H - cx_new))
        r_new = np.minimum(r_new, lo)
        r_new = np.maximum(r_new, 1e-6)
        
        x_new = np.column_stack([cx_new, cy_new, r_new])
        s_new = _evaluate(x_new, W, H)
        
        if s_new is not None:
            if s_new > best_sum:
                best_sum = s_new
                best_cx = cx_new.copy()
                best_cy = cy_new.copy()
                best_r = r_new.copy()
            if s_new >= current_sum:
                cx, cy, r = cx_new, cy_new, r_new
                current_sum = s_new
            else:
                delta = current_sum - s_new
                if random.random() < np.exp(-delta / max(T, 1e-8)):
                    cx, cy, r = cx_new, cy_new, r_new
                    current_sum = s_new
    
    return np.column_stack([best_cx, best_cy, best_r]), best_sum

def _local_refine(x0, W, H, max_iter=80):
    """Fine-tuning gradient descent with very small steps."""
    n = len(x0)
    x = x0.copy()
    best_x = x.copy()
    best_sum = _evaluate(x, W, H) or 0.0
    
    lr = 0.005
    for it in range(max_iter):
        if time.time() > DEADLINE - 3:
            break
        
        obj, grad = _penalty_and_gradient(x, W, H, penalty_weight=200.0)
        x_new = x - lr * grad
        x_new[:, 0] = np.clip(x_new[:, 0], 0.0, W)
        x_new[:, 1] = np.clip(x_new[:, 1], 0.0, H)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        lo = np.minimum(np.minimum(x_new[:, 0], x_new[:, 1]), np.minimum(W - x_new[:, 0], H - x_new[:, 1]))
        x_new[:, 2] = np.minimum(x_new[:, 2], lo)
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-6)
        
        s_new = _evaluate(x_new, W, H)
        if s_new is not None and s_new > best_sum:
            best_sum = s_new
            best_x = x_new.copy()
        
        obj_new, _ = _penalty_and_gradient(x_new, W, H, penalty_weight=200.0)
        if obj_new < obj:
            x = x_new
        else:
            lr *= 0.85
        
        lr = max(lr, 1e-6)
    
    return best_x, best_sum

def _grid_layout(kx, ky, W, H, shrink=0.95):
    """Build a kx*ky grid layout (pad/truncate to N)."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _random_layout(W, H):
    """Random feasible-ish layout."""
    pts = []
    for _ in range(N):
        cx = random.uniform(0.1 * W, 0.9 * W)
        cy = random.uniform(0.1 * H, 0.9 * H)
        r = min(cx, cy, W - cx, H - cy) * 0.4
        r = max(r, 1e-4)
        pts.append([cx, cy, r])
    return np.array(pts)

def _honeycomb_layout(W, H):
    """Hexagonal/honeycomb-like layout for better packing density."""
    pts = []
    base_r = min(W, H) / 8.0
    rows = int(H / (base_r * 1.732)) + 2
    cols = int(W / (base_r * 1.5)) + 2
    
    for row in range(rows):
        y = row * base_r * 1.732
        if y > H:
            break
        offset = (base_r * 1.5) if (row % 2 == 1) else 0
        for col in range(cols):
            x = col * base_r * 1.5 + offset
            if x > W:
                break
            r = min(x, y, W - x, H - y) * 0.9
            r = max(r, 1e-4)
            pts.append([x, y, r])
    
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    def time_ok():
        return time.time() < DEADLINE - 3
    
    random.seed(42)
    np.random.seed(42)
    
    candidates = []
    
    Ws = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    
    for W in Ws:
        H = 2.0 - W
        if H < 0.3 or W < 0.3:
            continue
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.95)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
        
        h = _honeycomb_layout(W, H)
        s_h = _evaluate(h, W, H)
        if s_h is not None:
            candidates.append((s_h, W, H, h))
        
        for _ in range(2):
            g = _random_layout(W, H)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
    
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 1: SLSQP on top candidates
    top_k = min(8, len(candidates))
    for idx in range(top_k):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        x_slsqp, s_slsqp = _slsqp_optimize(W, H, g0, maxiter=150)
        if s_slsqp > best_sum:
            best_sum = s_slsqp
            best_arr = x_slsqp
            best_W = W
            best_H = H
    
    # Phase 2: Gradient descent on remaining candidates
    for idx in range(top_k, min(16, len(candidates))):
        if not time_ok():
            break
        s0, W, H, g0 = candidates[idx]
        x_gd, s_gd = _gradient_descent(g0, W, H, max_iter=200, lr=0.01, penalty_weight=50.0)
        if s_gd > best_sum:
            best_sum = s_gd
            best_arr = x_gd
            best_W = W
            best_H = H
    
    # Phase 3: SA exploration on best
    if time_ok() and best_arr is not None:
        x_sa, s_sa = _sa_exploration(best_arr, best_W, best_H, steps=1200, T0=0.05, T1=0.001)
        if s_sa > best_sum:
            best_sum = s_sa
            best_arr = x_sa
    
    # Phase 4: Local refinement
    if time_ok() and best_arr is not None:
        x_ref, s_ref = _local_refine(best_arr, best_W, best_H, max_iter=60)
        if s_ref > best_sum:
            best_sum = s_ref
            best_arr = x_ref
    
    # Phase 5: More SA + GD cycles if time remains
    if time_ok() and best_arr is not None:
        remaining = DEADLINE - time.time()
        n_extra = min(4, int(remaining // 12))
        for _ in range(n_extra):
            if not time_ok():
                break
            x0 = best_arr + np.random.randn(*best_arr.shape) * 0.02
            x0[:, 0] = np.clip(x0[:, 0], 0.0, best_W)
            x0[:, 1] = np.clip(x0[:, 1], 0.0, best_H)
            x0[:, 2] = np.maximum(x0[:, 2], 1e-4)
            x_sa, s_sa = _sa_exploration(x0, best_W, best_H, steps=600, T0=0.04, T1=0.001)
            if s_sa > best_sum:
                best_sum = s_sa
                best_arr = x_sa
            if time_ok():
                x_ref, s_ref = _local_refine(best_arr, best_W, best_H, max_iter=40)
                if s_ref > best_sum:
                    best_sum = s_ref
                    best_arr = x_ref
    
    # Final local refinement
    if time_ok() and best_arr is not None:
        x_ref, s_ref = _local_refine(best_arr, best_W, best_H, max_iter=40)
        if s_ref > best_sum:
            best_sum = s_ref
            best_arr = x_ref
    
    # Fallback
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.95)
        best_arr = g
        best_W, best_H = W, H
    
    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return out
