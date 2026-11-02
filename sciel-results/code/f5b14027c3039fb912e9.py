"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Strategy:
 - Fixed-aspect-ratio smooth surrogate optimization with L-BFGS-B
 - Multiple aspect ratios with joint refinement
 - Simulated annealing refinement
 - Final high-precision polish pass
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _softplus_vec(x, beta=10.0):
    """Vectorized numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 1.0 / beta
    out[mask] = x[mask]
    xm = x[~mask]
    out[~mask] = np.log1p(np.exp(beta * xm)) / beta
    return out


def _softplus_grad_vec(x, beta=10.0):
    """Vectorized derivative of softplus: sigmoid(beta*x)."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * x))


def _compute_overlap_penalty(xs, ys, rs, beta=10.0):
    """Smooth overlap penalty using softplus surrogate (vectorized)."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    iu = np.triu_indices(N, k=1)
    vals = val[iu]
    penalty = np.sum(_softplus_vec(vals, beta))
    return penalty


def _compute_overlap_penalty_grad(xs, ys, rs, beta=10.0):
    """Gradient of overlap penalty w.r.t. xs, ys, rs."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = rs[:, None] + rs[None, :]
    val = min_dist - dist
    
    sig = _softplus_grad_vec(val, beta)
    
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    iu = np.triu_indices(N, k=1)
    i_idx, j_idx = iu
    dx_ij = dx[i_idx, j_idx]
    dy_ij = dy[i_idx, j_idx]
    dist_ij = dist[i_idx, j_idx]
    sig_ij = sig[i_idx, j_idx]
    
    inv_dist = np.where(dist_ij > 1e-10, 1.0 / dist_ij, 0.0)
    
    np.add.at(gx, i_idx, -sig_ij * dx_ij * inv_dist)
    np.add.at(gx, j_idx, sig_ij * dx_ij * inv_dist)
    
    np.add.at(gy, i_idx, -sig_ij * dy_ij * inv_dist)
    np.add.at(gy, j_idx, sig_ij * dy_ij * inv_dist)
    
    np.add.at(gr, i_idx, sig_ij)
    np.add.at(gr, j_idx, sig_ij)
    
    return gx, gy, gr


def _boundary_penalty(xs, ys, rs, w, h):
    """Squared boundary violation penalty (vectorized)."""
    pen = 0.0
    viol_l = np.maximum(rs - xs, 0.0)
    pen += np.sum(viol_l ** 2)
    viol_r = np.maximum(xs + rs - w, 0.0)
    pen += np.sum(viol_r ** 2)
    viol_b = np.maximum(rs - ys, 0.0)
    pen += np.sum(viol_b ** 2)
    viol_t = np.maximum(ys + rs - h, 0.0)
    pen += np.sum(viol_t ** 2)
    return pen


def _boundary_penalty_grad(xs, ys, rs, w, h):
    """Gradient of boundary penalty w.r.t. xs, ys, rs."""
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    viol_l = np.maximum(rs - xs, 0.0)
    mask_l = viol_l > 0
    gx[mask_l] -= 2.0 * viol_l[mask_l]
    gr[mask_l] += 2.0 * viol_l[mask_l]
    
    viol_r = np.maximum(xs + rs - w, 0.0)
    mask_r = viol_r > 0
    gx[mask_r] += 2.0 * viol_r[mask_r]
    gr[mask_r] += 2.0 * viol_r[mask_r]
    
    viol_b = np.maximum(rs - ys, 0.0)
    mask_b = viol_b > 0
    gy[mask_b] -= 2.0 * viol_b[mask_b]
    gr[mask_b] += 2.0 * viol_b[mask_b]
    
    viol_t = np.maximum(ys + rs - h, 0.0)
    mask_t = viol_t > 0
    gy[mask_t] += 2.0 * viol_t[mask_t]
    gr[mask_t] += 2.0 * viol_t[mask_t]
    
    return gx, gy, gr


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
    
    overlap_pen = _compute_overlap_penalty(xs, ys, rs, beta)
    boundary_pen = _boundary_penalty(xs, ys, rs, w, h)
    
    return overlap_pen + boundary_pen - np.sum(rs)


def _objective_grad(params, w, h, beta=10.0):
    """Gradient of objective."""
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    if np.any(rs <= 0):
        grad = np.zeros_like(params)
        grad[2::3] = 2.0 * np.minimum(rs, 0)
        return grad
    
    gx_o, gy_o, gr_o = _compute_overlap_penalty_grad(xs, ys, rs, beta)
    gx_b, gy_b, gr_b = _boundary_penalty_grad(xs, ys, rs, w, h)
    
    gx = gx_o + gx_b
    gy = gy_o + gy_b
    gr = gr_o + gr_b - 1.0
    
    grad = np.zeros_like(params)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return grad


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


def _sa_optimize(w, h, x0, max_iters=5000, seed=42, deadline=None):
    """Simulated annealing for fixed w, h with incremental cost update."""
    xs = x0[0::3].copy()
    ys = x0[1::3].copy()
    rs = x0[2::3].copy()
    
    r_max = min(w, h) / 2.0
    T = 0.08
    cooling = 0.9995
    beta = 10.0
    
    others_idx = [np.array([i for i in range(N) if i != idx]) for idx in range(N)]
    
    current_cost = _compute_overlap_penalty(xs, ys, rs, beta=beta) - np.sum(rs)
    
    rng = np.random.default_rng(seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        others = others_idx[idx]
        old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
        
        scale = T * 0.4
        dx_move = rng.normal(0, scale)
        dy_move = rng.normal(0, scale)
        dr_move = rng.normal(0, scale * 0.3)
        
        new_r = np.clip(old_r + dr_move, 1e-4, r_max)
        new_x = np.clip(old_x + dx_move, new_r, w - new_r)
        new_y = np.clip(old_y + dy_move, new_r, h - new_r)
        
        dx_old = old_x - xs[others]
        dy_old = old_y - ys[others]
        dist_old = np.sqrt(dx_old * dx_old + dy_old * dy_old + 1e-12)
        min_old = old_r + rs[others]
        vals_old = min_old - dist_old
        cost_old_idx = np.sum(_softplus_vec(vals_old, beta))
        
        xs[idx] = new_x
        ys[idx] = new_y
        rs[idx] = new_r
        
        dx_new = new_x - xs[others]
        dy_new = new_y - ys[others]
        dist_new = np.sqrt(dx_new * dx_new + dy_new * dy_new + 1e-12)
        min_new = new_r + rs[others]
        vals_new = min_new - dist_new
        cost_new_idx = np.sum(_softplus_vec(vals_new, beta))
        
        delta = (cost_new_idx - cost_old_idx) - (new_r - old_r)
        
        if delta < 0 or (T > 1e-7 and rng.random() < np.exp(-delta / T)):
            current_cost += delta
        else:
            xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
        
        T *= cooling
        if T < 0.001:
            T = 0.001
    
    x_final = np.zeros(3 * N)
    x_final[0::3] = xs
    x_final[1::3] = ys
    x_final[2::3] = rs
    return x_final


def _grid_initial(w, h, seed=0):
    """Create grid arrangement of 21 circles with optional jitter."""
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
    
    if seed > 0:
        rng = np.random.default_rng(seed)
        for i in range(N):
            x0[3 * i] += rng.normal(0, 0.02)
            x0[3 * i + 1] += rng.normal(0, 0.02)
            x0[3 * i + 2] *= (1.0 + rng.normal(0, 0.1))
            x0[3 * i] = np.clip(x0[3 * i], x0[3 * i + 2], w - x0[3 * i + 2])
            x0[3 * i + 1] = np.clip(x0[3 * i + 1], x0[3 * i + 2], h - x0[3 * i + 2])
            x0[3 * i + 2] = np.clip(x0[3 * i + 2], 1e-4, min(w, h) / 2.0)
    
    return x0


def _joint_aspect_optimize(x0, w, h, beta=10.0, maxiter=200, deadline=None):
    """Optimize positions AND aspect ratio jointly."""
    n_params = 3 * N + 1
    
    def _obj(p):
        wv = p[-1]
        hv = 2.0 - wv
        xs = p[0::3]
        ys = p[1::3]
        rs = p[2::3]
        if np.any(rs <= 0) or wv <= 0.01 or hv <= 0.01:
            return 1e10
        op = _compute_overlap_penalty(xs, ys, rs, beta)
        bp = _boundary_penalty(xs, ys, rs, wv, hv)
        return op + bp - np.sum(rs)
    
    def _obj_grad(p):
        wv = p[-1]
        hv = 2.0 - wv
        xs = p[0::3]
        ys = p[1::3]
        rs = p[2::3]
        if np.any(rs <= 0) or wv <= 0.01 or hv <= 0.01:
            return np.zeros(n_params)
        gx_o, gy_o, gr_o = _compute_overlap_penalty_grad(xs, ys, rs, beta)
        gx_b, gy_b, gr_b = _boundary_penalty_grad(xs, ys, rs, wv, hv)
        gx = gx_o + gx_b
        gy = gy_o + gy_b
        gr = gr_o + gr_b - 1.0
        viol_r = np.maximum(xs + rs - wv, 0.0)
        viol_t = np.maximum(ys + rs - hv, 0.0)
        gw = 2.0 * np.sum(viol_r) - 2.0 * np.sum(viol_t)
        
        grad = np.zeros(n_params)
        grad[0::3] = gx
        grad[1::3] = gy
        grad[2::3] = gr
        grad[-1] = gw
        return grad
    
    p0 = np.zeros(n_params)
    p0[0::3] = x0[0::3]
    p0[1::3] = x0[1::3]
    p0[2::3] = x0[2::3]
    p0[-1] = w
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1.5))
        bounds.append((0, 1.5))
        bounds.append((1e-4, 0.75))
    bounds.append((0.5, 1.5))
    
    try:
        res = minimize(_obj, p0, jac=_obj_grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
        return res.x
    except Exception:
        return p0


def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Focus on promising aspect ratios
    widths = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    
    best_x_for_warm = None
    best_w_for_warm = None
    
    for w in widths:
        if time.time() > deadline - 25.0:
            break
        h = 2.0 - w
        
        inits = [_grid_initial(w, h, seed=0)]
        if time.time() < deadline - 30.0:
            inits.append(_grid_initial(w, h, seed=1))
        
        for x0 in inits:
            if time.time() > deadline - 25.0:
                break
            
            bounds = []
            r_max = min(w, h) / 2.0
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-4, r_max))
            
            try:
                # First pass with moderate beta
                res = minimize(lambda p: _objective(p, w, h, beta=5.0), x0,
                               jac=lambda p: _objective_grad(p, w, h, beta=5.0),
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                if time.time() > deadline - 15.0:
                    break
                _check_and_update(res.x, w, h, best_state)
                
                # Second pass with higher beta
                if time.time() < deadline - 20.0:
                    res2 = minimize(lambda p: _objective(p, w, h, beta=20.0), res.x,
                                    jac=lambda p: _objective_grad(p, w, h, beta=20.0),
                                    method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
                    _check_and_update(res2.x, w, h, best_state)
                    if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                        best_x_for_warm = res2.x.copy()
                        best_w_for_warm = w
            except Exception:
                pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 30.0 and best_x_for_warm is not None:
            for sa_seed in [42, 123, 456]:
                if time.time() > deadline - 15.0:
                    break
                x_sa_start = best_x_for_warm.copy()
                if sa_seed > 42:
                    rng_sa = np.random.default_rng(sa_seed)
                    for i in range(N):
                        x_sa_start[3 * i] += rng_sa.normal(0, 0.01)
                        x_sa_start[3 * i + 1] += rng_sa.normal(0, 0.01)
                        x_sa_start[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa = _sa_optimize(w, h, x_sa_start, max_iters=3000, seed=sa_seed, deadline=deadline - 10.0)
                _check_and_update(x_sa, w, h, best_state)
                
                if time.time() < deadline - 12.0:
                    try:
                        res_pol = minimize(lambda p: _objective(p, w, h, beta=30.0), x_sa,
                                           jac=lambda p: _objective_grad(p, w, h, beta=30.0),
                                           method='L-BFGS-B', bounds=bounds,
                                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
                        _check_and_update(res_pol.x, w, h, best_state)
                    except Exception:
                        pass

    # Joint aspect-ratio optimization
    if time.time() < deadline - 20.0 and best_x_for_warm is not None and best_w_for_warm is not None:
        try:
            w_init = best_w_for_warm
            x_joint = _joint_aspect_optimize(best_x_for_warm, w_init, 2.0 - w_init, beta=10.0, maxiter=300, deadline=deadline - 10.0)
            w_opt = x_joint[-1]
            h_opt = 2.0 - w_opt
            x_fixed = x_joint[:-1]
            _check_and_update(x_fixed, w_opt, h_opt, best_state)
            
            if time.time() < deadline - 10.0:
                bounds_j = []
                r_max_j = min(w_opt, h_opt) / 2.0
                for i in range(N):
                    bounds_j.append((0, w_opt))
                    bounds_j.append((0, h_opt))
                    bounds_j.append((1e-4, r_max_j))
                res_jpol = minimize(lambda p: _objective(p, w_opt, h_opt, beta=50.0), x_fixed,
                                     jac=lambda p: _objective_grad(p, w_opt, h_opt, beta=50.0),
                                     method='L-BFGS-B', bounds=bounds_j,
                                     options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                _check_and_update(res_jpol.x, w_opt, h_opt, best_state)
        except Exception:
            pass

    # Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_state['result'] is not None:
        xs = best_state['result'][:, 0]
        ys = best_state['result'][:, 1]
        rs = best_state['result'][:, 2]
        
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
                res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0), x_polish,
                                      jac=lambda p: _objective_grad(p, w_polish, h_polish, beta=50.0),
                                      method='L-BFGS-B', bounds=bounds_polish,
                                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
                _check_and_update(res_polish.x, w_polish, h_polish, best_state)
                
                if time.time() < deadline - 3.0:
                    res_polish2 = minimize(lambda p: _objective(p, w_polish, h_polish, beta=100.0), res_polish.x,
                                           jac=lambda p: _objective_grad(p, w_polish, h_polish, beta=100.0),
                                           method='L-BFGS-B', bounds=bounds_polish,
                                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
                    _check_and_update(res_polish2.x, w_polish, h_polish, best_state)
            except Exception:
                pass
    
    # Fallback
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
    
    result = best_state['result'].copy()
    _scale_to_perimeter(result)
    return result
