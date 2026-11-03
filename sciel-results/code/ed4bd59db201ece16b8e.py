import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0, 
                    x + np.log1p(np.exp(-xb)) / beta, 
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0, 
                    1.0 / (1.0 + np.exp(-xb)), 
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_joint(state, beta=100.0):
    """
    State: [t, x1, y1, r1, ..., xN, yN, rN]
    t is the fraction of width: w = 2t, h = 2(1-t)
    """
    t = state[0]
    # Clip t to avoid degenerate rectangles
    t_eff = np.clip(t, 0.1, 0.9)
    w = 2.0 * t_eff
    h = 2.0 * (1.0 - t_eff)
    
    xs = state[1::3]
    ys = state[2::3]
    rs = state[3::3]
    
    obj = -np.sum(rs)
    grad = np.zeros_like(state)
    
    # 1. Boundary Constraints
    # Left: x - r >= 0 => v = r - x
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[1::3] -= g1
    grad[3::3] += g1

    # Right: x + r <= w => v = x + r - w
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[1::3] += g2
    grad[3::3] += g2
    # Gradient w.r.t t: d/dt (x + r - 2t) = -2
    # So grad[t] += sum(g2 * (-2)) ? 
    # Wait, v2 = x + r - w(t). dv2/dt = -dw/dt = -2.
    # d obj / dt += sum( g2 * (-2) )
    grad[0] += -2.0 * np.sum(g2)

    # Bottom: y - r >= 0 => v = r - y
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[2::3] -= g3
    grad[3::3] += g3

    # Top: y + r <= h => v = y + r - h
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[2::3] += g4
    grad[3::3] += g4
    # Gradient w.r.t t: d/dt (y + r - 2(1-t)) = d/dt (y + r - 2 + 2t) = 2
    # d obj / dt += sum( g4 * 2 )
    grad[0] += 2.0 * np.sum(g4)

    # Positive Radius: r > 0 => v = -r
    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[3::3] -= g5

    # 2. Pairwise Non-overlap
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    v = r_sum - d
    
    iu = np.triu_indices(N, k=1)
    v_upper = v[iu]
    d_upper = d[iu]
    dx_upper = dx[iu]
    dy_upper = dy[iu]
    
    pen = _softplus(v_upper, beta)
    obj += np.sum(pen)
    g = _softplus_grad(v_upper, beta)
    
    idx_i = iu[0]
    idx_j = iu[1]
    
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        gk = g[k]
        dk = d_upper[k]
        dxk = dx_upper[k]
        dyk = dy_upper[k]
        
        # Grad w.r.t x_i, y_i, r_i
        grad[1 + 3*i]   += -gk * dxk / dk
        grad[2 + 3*i]   += -gk * dyk / dk
        grad[3 + 3*i]   +=  gk
        
        # Grad w.r.t x_j, y_j, r_j
        grad[1 + 3*j]   +=  gk * dxk / dk
        grad[2 + 3*j]   +=  gk * dyk / dk
        grad[3 + 3*j]   +=  gk

    return obj, grad

def _strictly_valid(circles, w, h, tol=1e-8):
    n = circles.shape[0]
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx * dx + dy * dy
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    iu = np.triu_indices(n, k=1)
    d2_upper = d2[iu]
    r_sum_upper = r_sum[iu]
    if np.any(d2_upper < (r_sum_upper - tol) ** 2):
        return False
    return True

def _hex_init(t):
    """Generate hex-packed initial configuration for a given t."""
    w = 2.0 * t
    h = 2.0 * (1.0 - t)
    
    # Estimate radius
    # Area of rectangle = 2
    # Area of circles approx N * pi * r^2
    # Packing density ~ 0.9
    r_est = np.sqrt(0.9 * 2.0 / (N * np.pi))
    
    circles = []
    # Try to fit hex lattice
    r = r_est
    for attempt in range(10):
        circles = []
        y = r
        row_idx = 0
        while y <= h - r + 1e-9:
            x_offset = 0.0 if row_idx % 2 == 0 else r
            x = r + x_offset
            while x <= w - r + 1e-9:
                circles.append([x, y, r])
                x += 2.0 * r
            y += np.sqrt(3) * r
            row_idx += 1
        
        if len(circles) >= N:
            break
        r *= 0.95
        
    if len(circles) < N:
        # Fallback grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
            
    # Take first N
    circles = circles[:N]
    return np.array(circles)

def _objective_fixed_rect(x, w, h, beta=100.0):
    """
    Objective for fixed w, h.
    x: [x1, y1, r1, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # Boundary
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # Pairwise
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d = np.sqrt(dx * dx + dy * dy + 1e-12)
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    v = r_sum - d
    iu = np.triu_indices(N, k=1)
    v_upper = v[iu]
    d_upper = d[iu]
    dx_upper = dx[iu]
    dy_upper = dy[iu]
    
    pen = _softplus(v_upper, beta)
    obj += np.sum(pen)
    g = _softplus_grad(v_upper, beta)
    
    idx_i = iu[0]
    idx_j = iu[1]
    
    for k in range(len(idx_i)):
        i = idx_i[k]
        j = idx_j[k]
        gk = g[k]
        dk = d_upper[k]
        dxk = dx_upper[k]
        dyk = dy_upper[k]
        
        grad[3*i]   += -gk * dxk / dk
        grad[3*i+1] += -gk * dyk / dk
        grad[3*i+2] +=  gk
        
        grad[3*j]   +=  gk * dxk / dk
        grad[3*j+1] +=  gk * dyk / dk
        grad[3*j+2] +=  gk

    # Gradient for -sum(r)
    grad[2::3] += -1.0

    return obj, grad

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None
    best_t = 0.5

    # Initial aspect ratios to explore
    # t is w/2, so w = 2t. w+h=2.
    # t=0.5 -> w=1, h=1
    # t=0.4 -> w=0.8, h=1.2
    # t=0.6 -> w=1.2, h=0.8
    initial_ts = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65]

    for t_init in initial_ts:
        if time.time() > deadline - 10:
            break
            
        # Generate initial circles
        circles_init = _hex_init(t_init)
        
        # Construct initial state: [t, x1, y1, r1, ...]
        state = np.zeros(1 + 3*N)
        state[0] = t_init
        state[1::3] = circles_init[:, 0]
        state[2::3] = circles_init[:, 1]
        state[3::3] = circles_init[:, 2]
        
        # Multi-phase optimization
        # Phases: (beta, max_iter)
        phases = [
            (10.0, 100),
            (50.0, 100),
            (200.0, 100),
            (500.0, 100),
            (1000.0, 100)
        ]
        
        current_state = state
        for beta, max_iter in phases:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(
                    _objective_joint,
                    current_state,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                current_state = res.x
            except Exception:
                pass
        
        # Extract result
        t_opt = np.clip(current_state[0], 0.1, 0.9)
        w_opt = 2.0 * t_opt
        h_opt = 2.0 * (1.0 - t_opt)
        
        circles_opt = np.zeros((N, 3))
        circles_opt[:, 0] = current_state[1::3]
        circles_opt[:, 1] = current_state[2::3]
        circles_opt[:, 2] = current_state[3::3]
        
        # Clamp
        circles_opt[:, 0] = np.clip(circles_opt[:, 0], 0, w_opt)
        circles_opt[:, 1] = np.clip(circles_opt[:, 1], 0, h_opt)
        circles_opt[:, 2] = np.clip(circles_opt[:, 2], 1e-9, min(w_opt, h_opt)/2)
        
        if _strictly_valid(circles_opt, w_opt, h_opt):
            val = float(np.sum(circles_opt[:, 2]))
            if val > best_val:
                best_val = val
                best_result = circles_opt.copy()
                best_t = t_opt

    # Final Polish with High Beta on the best found configuration
    if best_result is not None and time.time() < deadline - 10:
        w_best = 2.0 * best_t
        h_best = 2.0 * (1.0 - best_t)
        
        x_init = best_result.flatten()
        
        # High beta polish phases
        polish_phases = [
            (2000.0, 100),
            (5000.0, 100)
        ]
        
        current_x = x_init
        for beta, max_iter in polish_phases:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    _objective_fixed_rect,
                    current_x,
                    args=(w_best, h_best, beta),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
                )
                current_x = res.x
            except Exception:
                pass
        
        c_polished = current_x.reshape(-1, 3)
        c_polished[:, 0] = np.clip(c_polished[:, 0], 0, w_best)
        c_polished[:, 1] = np.clip(c_polished[:, 1], 0, h_best)
        c_polished[:, 2] = np.clip(c_polished[:, 2], 1e-9, min(w_best, h_best)/2)
        
        if _strictly_valid(c_polished, w_best, h_best):
            val_polished = float(np.sum(c_polished[:, 2]))
            if val_polished > best_val:
                best_val = val_polished
                best_result = c_polished.copy()

    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        a = 5
        b = 5
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = 0.15
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best_result = np.array(circles)

    return best_result
