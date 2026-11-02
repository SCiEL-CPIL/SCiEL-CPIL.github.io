"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - Analytical gradients for L-BFGS-B to improve speed and convergence
 - Multiple aspect ratios search
 - Simulated annealing refinement
 - Final high-precision polish pass
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus(x, beta=10.0):
    """Numerically stable softplus."""
    # For large x, softplus(x) ~ x
    # For small x, softplus(x) ~ log(1 + exp(beta*x))/beta
    # To avoid overflow in exp:
    if x > 1.0 / beta:
        return x
    return np.log1p(np.exp(beta * x)) / beta


def _softplus_grad(x, beta=10.0):
    """Derivative of softplus: sigmoid(beta * x)."""
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For numerical stability:
    # if z > 0: 1 / (1 + exp(-z))
    # if z <= 0: exp(z) / (1 + exp(z))
    z = beta * x
    if z > 0:
        return 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        return ez / (1.0 + ez)


def _objective(params, w, h, beta=10.0):
    """
    Objective: minimize penalty - sum(r).
    params = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs <= 0):
        return 1e10 + np.sum(np.minimum(rs, 0) ** 2)
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            val = min_dist - dist
            overlap_pen += _softplus(val, beta)
            
    # Boundary penalty (squared)
    boundary_pen = 0.0
    for i in range(N):
        if xs[i] < rs[i]:
            boundary_pen += (rs[i] - xs[i]) ** 2
        if xs[i] > w - rs[i]:
            boundary_pen += (xs[i] - (w - rs[i])) ** 2
        if ys[i] < rs[i]:
            boundary_pen += (rs[i] - ys[i]) ** 2
        if ys[i] > h - rs[i]:
            boundary_pen += (ys[i] - (h - rs[i])) ** 2
            
    return overlap_pen + boundary_pen - np.sum(rs)


def _objective_grad(params, w, h, beta=10.0):
    """
    Analytical gradient of _objective.
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    grad = np.zeros(3 * N)
    
    if np.any(rs <= 0):
        # Simple gradient for the penalty term in invalid region
        grad[2::3] = 2.0 * np.minimum(rs, 0)
        return grad

    # 1. Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # 2. Gradient of Overlap Penalty
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            
            min_dist = rs[i] + rs[j]
            val = min_dist - dist
            sp_grad = _softplus_grad(val, beta)
            
            # d(softplus(val))/d(val) = sp_grad
            # d(val)/d(x_i) = -dx/dist
            # d(val)/d(y_i) = -dy/dist
            # d(val)/d(r_i) = 1
            
            # Contribution to x_i
            grad[3*i] += sp_grad * (-dx / dist)
            # Contribution to y_i
            grad[3*i + 1] += sp_grad * (-dy / dist)
            # Contribution to r_i
            grad[3*i + 2] += sp_grad
            
            # Contribution to x_j
            grad[3*j] += sp_grad * (dx / dist)
            # Contribution to y_j
            grad[3*j + 1] += sp_grad * (dy / dist)
            # Contribution to r_j
            grad[3*j + 2] += sp_grad

    # 3. Gradient of Boundary Penalty
    for i in range(N):
        # x boundaries
        if xs[i] < rs[i]:
            # pen = (r_i - x_i)^2
            # d/dx_i = -2(r_i - x_i) = 2(x_i - r_i)
            # d/dr_i = 2(r_i - x_i)
            grad[3*i] += 2.0 * (xs[i] - rs[i])
            grad[3*i + 2] += 2.0 * (rs[i] - xs[i])
        if xs[i] > w - rs[i]:
            # pen = (x_i - (w - r_i))^2
            # d/dx_i = 2(x_i - w + r_i)
            # d/dr_i = -2(x_i - w + r_i)
            grad[3*i] += 2.0 * (xs[i] - w + rs[i])
            grad[3*i + 2] += -2.0 * (xs[i] - w + rs[i])
            
        # y boundaries
        if ys[i] < rs[i]:
            grad[3*i + 1] += 2.0 * (ys[i] - rs[i])
            grad[3*i + 2] += 2.0 * (rs[i] - ys[i])
        if ys[i] > h - rs[i]:
            grad[3*i + 1] += 2.0 * (ys[i] - h + rs[i])
            grad[3*i + 2] += -2.0 * (ys[i] - h + rs[i])

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


def _check_and_update(params, w, h, best_state):
    """Check validity and update best if improved."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs < 1e-9):
        return
    if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
        return
    if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
        return
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < (min_dist - 1e-7) ** 2:
                return
    
    val = np.sum(rs)
    if val > best_state['value']:
        best_state['value'] = val
        best_state['result'] = np.zeros((N, 3))
        best_state['result'][:, 0] = xs
        best_state['result'][:, 1] = ys
        best_state['result'][:, 2] = rs


def _sa_optimize(w, h, x0, max_iters=2000, seed=42, deadline=None):
    """Simulated annealing for fixed w, h."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    r_max = min(w, h) / 2.0
    T = 0.05
    cooling = 0.999
    current_cost = _objective(x0, w, h, beta=10.0)
    
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
        
        scale = T * 0.3
        xs[idx] += rng.normal(0, scale)
        ys[idx] += rng.normal(0, scale)
        rs[idx] += rng.normal(0, scale * 0.5)
        
        xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
        ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
        rs[idx] = np.clip(rs[idx], 1e-4, r_max)
        
        x_temp = np.zeros(3*N)
        x_temp[0::3] = xs
        x_temp[1::3] = ys
        x_temp[2::3] = rs
        
        new_cost = _objective(x_temp, w, h, beta=10.0)
        delta = new_cost - current_cost
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_cost = new_cost
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
        
        T *= cooling
    
    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def _grid_initial(w, h):
    """Create grid arrangement of 21 circles."""
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    
    x0 = np.zeros(3 * N)
    for i in range(N):
        col = i % a
        row = i // a
        x0[3 * i] = (col + 0.5) * w / a
        x0[3 * i + 1] = (row + 0.5) * h / b
        x0[3 * i + 2] = min(w / a, h / b) * 0.45
    
    return x0


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Try multiple aspect ratios
    widths = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    best_x_for_warm = None
    
    for w in widths:
        if time.time() > deadline - 20.0:
            break
        h = 2.0 - w
        
        # Initialize with grid
        x0 = _grid_initial(w, h)
        
        # L-BFGS-B optimization with smooth objective and analytical gradient
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta
            res = minimize(lambda p: _objective(p, w, h, beta=5.0),
                           lambda p: _objective_grad(p, w, h, beta=5.0),
                           x0, method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline - 10.0:
                break
            _check_and_update(res.x, w, h, best_state)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 15.0:
                res2 = minimize(lambda p: _objective(p, w, h, beta=20.0),
                                lambda p: _objective_grad(p, w, h, beta=20.0),
                                res.x, method='L-BFGS-B', bounds=bounds, jac=True,
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
                _check_and_update(res2.x, w, h, best_state)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
        except Exception:
            pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 25.0:
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_sa = _sa_optimize(w, h, x_sa_start, max_iters=1500, seed=42, deadline=deadline - 10.0)
            _check_and_update(x_sa, w, h, best_state)
            
            # Second SA pass with different seed
            if time.time() < deadline - 15.0:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                for i in range(N):
                    x_sa2[3 * i] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 1] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa2 = _sa_optimize(w, h, x_sa2, max_iters=1000, seed=123, deadline=deadline - 10.0)
                _check_and_update(x_sa2, w, h, best_state)

    # Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_state['result'] is not None:
        # Extract best parameters
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]
        
        # Determine w, h from bounding box
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        w_polish = bw
        h_polish = bh
        
        if w_polish > 0 and h_polish > 0:
            x_polish = np.zeros(3 * N)
            x_polish[0::3] = xs - min_x
            x_polish[1::3] = ys - min_y
            x_polish[2::3] = rs
            
            bounds_polish = []
            r_max_polish = min(w_polish, h_polish) / 2.0
            for i in range(N):
                bounds_polish.append((0, w_polish))
                bounds_polish.append((0, h_polish))
                bounds_polish.append((1e-4, r_max_polish))
            
            try:
                res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0),
                                      lambda p: _objective_grad(p, w_polish, h_polish, beta=50.0),
                                      x_polish, method='L-BFGS-B', bounds=bounds_polish, jac=True,
                                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res_polish.x, w_polish, h_polish, best_state)
            except Exception:
                pass
    
    # Fallback if no valid result found
    if best_state['result'] is None or best_state['value'] <= 0:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros((N, 3))
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[idx, 0] = (i + 0.5) * w / a
                best_result[idx, 1] = (j + 0.5) * h / b
                best_result[idx, 2] = r
        _scale_to_perimeter(best_result)
        return best_result
    
    # Scale to ensure perimeter <= 4
    result = best_state['result'].copy()
    _scale_to_perimeter(result)
    return result
