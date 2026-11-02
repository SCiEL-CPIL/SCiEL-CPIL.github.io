import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay
import time

N = 21
TIME_LIMIT = 148.0  # Safety margin before 150s kill

def _softplus(x, beta=100.0):
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    return 1.0 / (1.0 + np.exp(-beta * x))

def _objective_and_grad(x, w, h, beta=100.0):
    """
    Objective: maximize sum(r) subject to circle packing constraints.
    Uses softplus penalties for constraints.
    x is shape (3*N,): [x0,y0,r0, x1,y1,r1, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)

    # Boundary penalties
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

    # Pairwise overlap penalties
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        obj += np.sum(_softplus(v, beta))
        g = _softplus_grad(v, beta)

        grad[3 * i] += -np.sum(g * dx / d)
        grad[3 * i + 1] += -np.sum(g * dy / d)
        grad[3 * i + 2] += np.sum(g)

        idx = np.arange(i + 1, N)
        grad[3 * idx] += g * dx / d
        grad[3 * idx + 1] += g * dy / d
        grad[3 * idx + 2] += g

    grad[2::3] += -1.0
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
    for i in range(n):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d2 = dx * dx + dy * dy
        need = rs[i] + rs[i + 1:] - tol
        if np.any(d2 < need * need):
            return False
    return True


def _solve_lp_radii(xs, ys, w, h):
    """
    Solve for maximum radii given fixed positions using LP.
    """
    n = N
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # All pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    
    # Boundary constraints
    for i in range(n):
        for bound_val in [xs[i], w-xs[i], ys[i], h-ys[i]]:
            row = np.zeros(n)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(max(0, bound_val))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    
    # Fallback: greedy radius assignment
    rs = np.zeros(n)
    for i in range(n):
        max_r = min(xs[i], w-xs[i], ys[i], h-ys[i])
        for j in range(n):
            if i != j:
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - rs[j])
        rs[i] = max(0, max_r)
    return rs


def _hex_init(w, h, seed):
    """Generate hex-packed initial configurations."""
    rng = np.random.RandomState(seed)
    row_configs = [
        [3, 4, 4, 4, 3, 3],
        [4, 4, 4, 4, 3, 2],
        [2, 4, 4, 4, 4, 3],
        [3, 4, 4, 4, 4, 2],
        [4, 3, 4, 4, 4, 2],
        [2, 3, 4, 4, 4, 4],
        [3, 3, 4, 4, 4, 3],
        [4, 4, 3, 4, 4, 2],
        [2, 4, 3, 4, 4, 4],
        [3, 4, 3, 4, 4, 3],
        [5, 4, 4, 4, 4],
        [4, 5, 4, 4, 4],
        [3, 5, 4, 4, 5],
        [4, 4, 5, 4, 4],
        [2, 5, 4, 5, 5],
        [5, 5, 5, 5, 1],
        [1, 5, 5, 5, 5],
        [3, 4, 5, 4, 5],
        [5, 4, 5, 4, 3],
        [4, 4, 4, 4, 4, 1],
        [1, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        est_r_h = w / (max_cols + 0.5)
        est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = (row_idx + 0.5) * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
    if best is None:
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
        circles = []
        for i in range(N):
            col = i % a
            row = i // a
            r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best = np.array(circles)
    return best


def _grid_init(w, h, seed):
    a = int(np.ceil(np.sqrt(N * w / h)))
    b = int(np.ceil(N / a))
    while a * b < N:
        b += 1
    circles = []
    for i in range(N):
        col = i % a
        row = i // a
        r = min(w / (a + 0.5), h / (b + 0.5)) * 0.9
        x = (col + 0.5) * w / a
        y = (row + 0.5) * h / b
        circles.append([x, y, r])
    return np.array(circles)


def _random_init(w, h, seed):
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles


def _force_directed(circles, w, h, n_iter=200):
    """Force-directed relaxation."""
    xs = circles[:, 0].copy()
    ys = circles[:, 1].copy()
    rs = _solve_lp_radii(xs, ys, w, h)
    
    for it in range(n_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    force = 0.1 * overlap
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
                elif dist <= 1e-10:
                    fx[i] += 0.01 * np.random.normal()
                    fy[i] += 0.01 * np.random.normal()
                    fx[j] -= 0.01 * np.random.normal()
                    fy[j] -= 0.01 * np.random.normal()
        
        for i in range(N):
            fx[i] += 0.001 * (w/2 - xs[i])
            fy[i] += 0.001 * (h/2 - ys[i])
            
            if xs[i] < rs[i]:
                fx[i] += 0.5 * (rs[i] - xs[i])
            if xs[i] > w - rs[i]:
                fx[i] -= 0.5 * (xs[i] - (w - rs[i]))
            if ys[i] < rs[i]:
                fy[i] += 0.5 * (rs[i] - ys[i])
            if ys[i] > h - rs[i]:
                fy[i] -= 0.5 * (ys[i] - (h - rs[i]))
        
        xs += 0.1 * fx
        ys += 0.1 * fy
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        rs = _solve_lp_radii(xs, ys, w, h)
    
    return np.column_stack([xs, ys, rs])


def construct():
    start_time = time.time()
    
    best_val = 0.0
    best_params = None
    best_w = 1.0
    best_h = 1.0
    
    rng = np.random.default_rng(42)
    
    # Search over aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.5, 0.9, 1.1, 0.75, 1.3, 0.55]
    
    for ar in aspect_ratios:
        if time.time() - start_time > TIME_LIMIT - 20:
            break
            
        a = 1.0 / ar  # h/w
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)
        
        # Generate multiple initializations
        inits = []
        for seed in range(4):
            circ = _hex_init(w, h, seed)
            inits.append(circ)
        for seed in range(2):
            circ = _grid_init(w, h, seed)
            inits.append(circ)
        for seed in range(2):
            circ = _random_init(w, h, seed)
            inits.append(circ)
        
        for params0 in inits:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            # Step 1: Force-directed relaxation
            try:
                params_fd = _force_directed(params0, w, h, n_iter=150)
            except Exception:
                params_fd = params0
            
            # Step 2: Gradient-based optimization with softplus
            x_flat = params_fd.flatten()
            bounds = [(0, w)] * N + [(0, h)] * N + [(0, min(w, h)/2)] * N
            
            try:
                res = minimize(_objective_and_grad, x_flat,
                              args=(w, h, 100.0),
                              jac=True,
                              method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.success:
                    new_circles = res.x.reshape(N, 3)
                    # Final LP solve for optimal radii
                    xs = new_circles[:, 0]
                    ys = new_circles[:, 1]
                    rs = _solve_lp_radii(xs, ys, w, h)
                    params_final = np.column_stack([xs, ys, rs])
                    
                    if _strictly_valid(params_final, w, h):
                        val = float(np.sum(rs))
                        if val > best_val:
                            best_val = val
                            best_params = params_final.copy()
                            best_w = w
                            best_h = h
            except Exception:
                pass
    
    # Final polish on best solution
    if best_params is not None and time.time() - start_time < TIME_LIMIT - 5:
        w = best_w
        h = best_h
        
        # Try more iterations of force-directed
        try:
            params_fd = _force_directed(best_params, w, h, n_iter=300)
            xs = params_fd[:, 0]
            ys = params_fd[:, 1]
            rs = _solve_lp_radii(xs, ys, w, h)
            params_final = np.column_stack([xs, ys, rs])
            
            if _strictly_valid(params_final, w, h):
                val = float(np.sum(rs))
                if val > best_val:
                    best_val = val
                    best_params = params_final.copy()
        except Exception:
            pass
        
        # Try gradient optimization again with higher beta
        x_flat = best_params.flatten()
        bounds = [(0, w)] * N + [(0, h)] * N + [(0, min(w, h)/2)] * N
        try:
            res = minimize(_objective_and_grad, x_flat,
                          args=(w, h, 200.0),
                          jac=True,
                          method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success:
                new_circles = res.x.reshape(N, 3)
                xs = new_circles[:, 0]
                ys = new_circles[:, 1]
                rs = _solve_lp_radii(xs, ys, w, h)
                params_final = np.column_stack([xs, ys, rs])
                
                if _strictly_valid(params_final, w, h):
                    val = float(np.sum(rs))
                    if val > best_val:
                        best_val = val
                        best_params = params_final.copy()
        except Exception:
            pass
    
    if best_params is not None:
        out = best_params.copy()
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
