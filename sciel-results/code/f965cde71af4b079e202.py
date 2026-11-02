"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.

Combined approach:
 - Smooth surrogate objective (softplus for non-overlap, squared for boundary)
 - Analytical gradients for L-BFGS-B
 - Delaunay-based LP radii solver for exact feasibility given centers
 - Multiple aspect ratios search
 - Simulated annealing refinement
 - Final high-precision polish pass
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21


def _softplus_vec(x, beta=10.0):
    """Numerically stable softplus, vectorized."""
    z = beta * x
    return np.where(z > 0, (z + np.log1p(np.exp(-z))) / beta, np.log1p(np.exp(z)) / beta)


def _sigmoid_vec(x, beta=10.0):
    """Derivative of softplus: sigmoid(beta * x), vectorized and stable."""
    z = beta * x
    return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


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
    
    # Overlap penalty (vectorized)
    ii, jj = np.triu_indices(N, k=1)
    dx = xs[ii] - xs[jj]
    dy = ys[ii] - ys[jj]
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.maximum(dist, 1e-12)
    min_dist = rs[ii] + rs[jj]
    val = min_dist - dist
    overlap_pen = np.sum(_softplus_vec(val, beta))
    
    # Boundary penalty (squared, vectorized)
    boundary_pen = 0.0
    boundary_pen += np.sum(np.maximum(rs - xs, 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(xs - (w - rs), 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(rs - ys, 0.0) ** 2)
    boundary_pen += np.sum(np.maximum(ys - (h - rs), 0.0) ** 2)
    
    return overlap_pen + boundary_pen - np.sum(rs)


def _objective_grad(params, w, h, beta=10.0):
    """
    Analytical gradient of _objective (vectorized).
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    grad = np.zeros(3 * N)
    
    if np.any(rs <= 0):
        grad[2::3] = 2.0 * np.minimum(rs, 0)
        return grad

    # 1. Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # 2. Gradient of Overlap Penalty (vectorized)
    ii, jj = np.triu_indices(N, k=1)
    dx = xs[ii] - xs[jj]
    dy = ys[ii] - ys[jj]
    dist = np.sqrt(dx * dx + dy * dy)
    dist_safe = np.maximum(dist, 1e-12)
    min_dist = rs[ii] + rs[jj]
    val = min_dist - dist_safe
    sp_grad = _sigmoid_vec(val, beta)
    
    inv = sp_grad / dist_safe
    np.add.at(grad, 3 * ii, -inv * dx)
    np.add.at(grad, 3 * ii + 1, -inv * dy)
    np.add.at(grad, 3 * ii + 2, sp_grad)
    np.add.at(grad, 3 * jj, inv * dx)
    np.add.at(grad, 3 * jj + 1, inv * dy)
    np.add.at(grad, 3 * jj + 2, sp_grad)

    # 3. Gradient of Boundary Penalty (vectorized)
    idx = np.arange(N)
    mask = xs < rs
    grad[3 * idx[mask]] += 2.0 * (xs[mask] - rs[mask])
    grad[3 * idx[mask] + 2] += 2.0 * (rs[mask] - xs[mask])
    mask = xs > w - rs
    grad[3 * idx[mask]] += 2.0 * (xs[mask] - w + rs[mask])
    grad[3 * idx[mask] + 2] += -2.0 * (xs[mask] - w + rs[mask])
    mask = ys < rs
    grad[3 * idx[mask] + 1] += 2.0 * (ys[mask] - rs[mask])
    grad[3 * idx[mask] + 2] += 2.0 * (rs[mask] - ys[mask])
    mask = ys > h - rs
    grad[3 * idx[mask] + 1] += 2.0 * (ys[mask] - h + rs[mask])
    grad[3 * idx[mask] + 2] += -2.0 * (ys[mask] - h + rs[mask])

    return grad


def _delaunay_edges(xs, ys):
    """Return list of (i, j) edges from Delaunay triangulation of centers."""
    points = np.column_stack((xs, ys))
    try:
        tri = Delaunay(points)
    except Exception:
        return None
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
    return list(edges)


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve LP for per-circle radii given fixed centers and rectangle.
    Maximize sum(r) s.t. r_i + r_j <= d_ij (Delaunay edges)
                          r_i <= min(x_i, w-x_i, y_i, h-y_i)
    """
    edges = _delaunay_edges(xs, ys)
    if edges is None:
        return None
    
    # Distance for each edge
    d = np.zeros(len(edges))
    for k, (i, j) in enumerate(edges):
        d[k] = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
    
    # Boundary limits
    r_max = np.minimum(np.minimum(xs, w - xs), np.minimum(ys, h - ys))
    r_max = np.maximum(r_max, 1e-8)
    
    # LP: maximize sum(r)
    # c = -1 for each r_i (minimize -sum(r))
    c = -np.ones(N)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # For each edge (i, j): r_i + r_j <= d_k
    A_rows = []
    b_vals = []
    for k, (i, j) in enumerate(edges):
        row = np.zeros(N)
        row[i] = 1.0
        row[j] = 1.0
        A_rows.append(row)
        b_vals.append(d[k])
    
    # Boundary: r_i <= r_max_i
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(r_max[i])
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    bounds = [(1e-8, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None


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


def _joint_aspect_optimize(xs, ys, rs, beta=10.0, max_iter=200, tol=1e-9, deadline=None):
    """
    Jointly optimize circle positions/radii AND rectangle aspect ratio.
    """
    w = 1.0
    h = 1.0
    xs = xs.copy()
    ys = ys.copy()
    rs = rs.copy()
    r_max = min(w, h) / 2.0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        bw = max_x - min_x
        bh = max_y - min_y
        if bw + bh < 1e-6:
            break
        s = 4.0 / (2.0 * (bw + bh))
        w_new = bw * s
        h_new = bh * s
        if abs(w_new - w) < tol and abs(h_new - h) < tol:
            w, h = w_new, h_new
            break
        w, h = w_new, h_new
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        xs = (xs - cx) * s + 0.5 * w
        ys = (ys - cy) * s + 0.5 * h
        rs = rs * s
        r_max = min(w, h) / 2.0
        params = np.zeros(3 * N)
        params[0::3] = xs
        params[1::3] = ys
        params[2::3] = rs
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        try:
            res = minimize(lambda p: _objective(p, w, h, beta=beta),
                           lambda p: _objective_grad(p, w, h, beta=beta),
                           params, method='L-BFGS-B', bounds=bounds, jac=True,
                           options={'maxiter': 60, 'ftol': 1e-10, 'gtol': 1e-10})
            xs = res.x[0::3]
            ys = res.x[1::3]
            rs = res.x[2::3]
        except Exception:
            pass

    return w, h, xs, ys, rs


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_state = {'value': -np.inf, 'result': None}
    
    # Try multiple aspect ratios
    widths = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    best_x_for_warm = None
    
    for w in widths:
        if time.time() > deadline - 25.0:
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
            if time.time() > deadline - 15.0:
                break
            _check_and_update(res.x, w, h, best_state)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 20.0:
                res2 = minimize(lambda p: _objective(p, w, h, beta=20.0),
                                lambda p: _objective_grad(p, w, h, beta=20.0),
                                res.x, method='L-BFGS-B', bounds=bounds, jac=True,
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
                _check_and_update(res2.x, w, h, best_state)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
        except Exception:
            pass
        
        # Delaunay LP radii refinement: fix centers, solve exact LP for radii
        if time.time() < deadline - 30.0 and best_x_for_warm is not None:
            try:
                xs_lp = best_x_for_warm[0::3].copy()
                ys_lp = best_x_for_warm[1::3].copy()
                # Clamp positions to valid range
                xs_lp = np.clip(xs_lp, 1e-4, w - 1e-4)
                ys_lp = np.clip(ys_lp, 1e-4, h - 1e-4)
                r_lp = _solve_lp_radii(xs_lp, ys_lp, w, h)
                if r_lp is not None:
                    x_lp = np.zeros(3 * N)
                    x_lp[0::3] = xs_lp
                    x_lp[1::3] = ys_lp
                    x_lp[2::3] = r_lp
                    _check_and_update(x_lp, w, h, best_state)
                    if np.sum(r_lp) > (best_x_for_warm is not None and np.sum(best_x_for_warm[2::3])) or best_x_for_warm is None:
                        if best_x_for_warm is None or np.sum(r_lp) > np.sum(best_x_for_warm[2::3]):
                            best_x_for_warm = x_lp.copy()
            except Exception:
                pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 35.0:
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else x0
            x_sa = _sa_optimize(w, h, x_sa_start, max_iters=3000, seed=42, deadline=deadline - 15.0)
            _check_and_update(x_sa, w, h, best_state)
            
            # Second SA pass with different seed
            if time.time() < deadline - 25.0:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                for i in range(N):
                    x_sa2[3 * i] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 1] += rng_sa.normal(0, 0.01)
                    x_sa2[3 * i + 2] += rng_sa.normal(0, 0.005)
                x_sa2 = _sa_optimize(w, h, x_sa2, max_iters=2000, seed=123, deadline=deadline - 15.0)
                _check_and_update(x_sa2, w, h, best_state)

    # Final high-precision polish pass
    if time.time() < deadline - 10.0 and best_state['result'] is not None:
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
                res_polish = minimize(lambda p: _objective(p, w_polish, h_polish, beta=50.0),
                                      lambda p: _objective_grad(p, w_polish, h_polish, beta=50.0),
                                      x_polish, method='L-BFGS-B', bounds=bounds_polish, jac=True,
                                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                _check_and_update(res_polish.x, w_polish, h_polish, best_state)
            except Exception:
                pass
            
            # Delaunay LP on polished result
            if time.time() < deadline - 5.0:
                try:
                    xs_f = res_polish.x[0::3].copy() if res_polish.success else x_polish[0::3].copy()
                    ys_f = res_polish.x[1::3].copy() if res_polish.success else x_polish[1::3].copy()
                    xs_f = np.clip(xs_f, 1e-4, w_polish - 1e-4)
                    ys_f = np.clip(ys_f, 1e-4, h_polish - 1e-4)
                    r_f = _solve_lp_radii(xs_f, ys_f, w_polish, h_polish)
                    if r_f is not None:
                        x_f = np.zeros(3 * N)
                        x_f[0::3] = xs_f
                        x_f[1::3] = ys_f
                        x_f[2::3] = r_f
                        _check_and_update(x_f, w_polish, h_polish, best_state)
                except Exception:
                    pass

        # Joint aspect-ratio refinement
        if time.time() < deadline - 3.0 and best_state['result'] is not None:
            xs = best_state['result'][:, 0]
            ys = best_state['result'][:, 1]
            rs = best_state['result'][:, 2]
            try:
                w_j, h_j, xs_j, ys_j, rs_j = _joint_aspect_optimize(
                    xs, ys, rs, beta=50.0, max_iter=120, deadline=deadline - 2.0)
                res_j = np.zeros((N, 3))
                res_j[:, 0] = xs_j
                res_j[:, 1] = ys_j
                res_j[:, 2] = rs_j
                min_x = np.min(xs_j - rs_j)
                min_y = np.min(ys_j - rs_j)
                res_j[:, 0] -= min_x
                res_j[:, 1] -= min_y
                ok = True
                if np.any(rs_j < 1e-9) or np.any(xs_j - rs_j < -1e-6) or np.any(xs_j + rs_j > w_j + 1e-6):
                    ok = False
                if np.any(ys_j - rs_j < -1e-6) or np.any(ys_j + rs_j > h_j + 1e-6):
                    ok = False
                if ok:
                    for i in range(N):
                        for j in range(i + 1, N):
                            d = np.sqrt((xs_j[i] - xs_j[j]) ** 2 + (ys_j[i] - ys_j[j]) ** 2)
                            if d < rs_j[i] + rs_j[j] - 1e-7:
                                ok = False
                                break
                if ok:
                    val_j = np.sum(rs_j)
                    if val_j > best_state['value']:
                        best_state['value'] = val_j
                        best_state['result'] = res_j.copy()
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
