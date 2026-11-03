import numpy as np
import time
import random
from scipy.optimize import minimize, linprog
from scipy.spatial import Delaunay

N = 21
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1+exp(beta*x))/beta"""
    xb = beta * x
    return np.where(xb > 0, 
                    x + np.log1p(np.exp(-xb)) / beta, 
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus: sigmoid(beta*x)"""
    xb = beta * x
    return np.where(xb > 0, 
                    1.0 / (1.0 + np.exp(-xb)), 
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_and_grad(x, w, h, beta=100.0, var_weight=0.0):
    """
    Objective: -sum(r) + penalties for violations.
    x: [x1, y1, r1, x2, y2, r2, ...]
    """
    c = x.reshape(-1, 3)
    xs = c[:, 0]
    ys = c[:, 1]
    rs = c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    
    # 1. Boundary Constraints
    # Left: x - r >= 0  => v = r - x
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[0::3] -= g1
    grad[2::3] += g1

    # Right: x + r <= w => v = x + r - w
    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[0::3] += g2
    grad[2::3] += g2

    # Bottom: y - r >= 0 => v = r - y
    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[1::3] -= g3
    grad[2::3] += g3

    # Top: y + r <= h => v = y + r - h
    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[1::3] += g4
    grad[2::3] += g4

    # Positive Radius: r > 0 => v = -r
    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[2::3] -= g5

    # 2. Pairwise Non-overlap Constraints
    # d_ij >= r_i + r_j
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

    # 3. Variance Penalty (Equalization Pressure)
    if var_weight > 0.0:
        r_mean = np.mean(rs)
        diff = rs - r_mean
        obj += var_weight * np.sum(diff * diff)
        grad[2::3] += 2.0 * var_weight * diff

    # 4. Objective gradient for -sum(r)
    grad[2::3] += -1.0

    return obj, grad

def _objective_joint(state, beta=100.0):
    """
    Joint objective: state = [t, x1, y1, r1, ..., xN, yN, rN]
    t in (0,1): w = 2t, h = 2(1-t), perimeter = 4
    """
    t = np.clip(state[0], 0.15, 0.85)
    w = 2.0 * t
    h = 2.0 * (1.0 - t)
    
    xs = state[1::3]
    ys = state[2::3]
    rs = state[3::3]
    
    obj = -np.sum(rs)
    grad = np.zeros_like(state)
    
    # Boundary penalties
    v1 = rs - xs
    obj += np.sum(_softplus(v1, beta))
    g1 = _softplus_grad(v1, beta)
    grad[1::3] -= g1
    grad[3::3] += g1

    v2 = xs + rs - w
    obj += np.sum(_softplus(v2, beta))
    g2 = _softplus_grad(v2, beta)
    grad[1::3] += g2
    grad[3::3] += g2

    v3 = rs - ys
    obj += np.sum(_softplus(v3, beta))
    g3 = _softplus_grad(v3, beta)
    grad[2::3] -= g3
    grad[3::3] += g3

    v4 = ys + rs - h
    obj += np.sum(_softplus(v4, beta))
    g4 = _softplus_grad(v4, beta)
    grad[2::3] += g4
    grad[3::3] += g4

    v5 = -rs
    obj += np.sum(_softplus(v5, beta))
    g5 = _softplus_grad(v5, beta)
    grad[3::3] -= g5

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
        
        grad[3*i+1]   += -gk * dxk / dk
        grad[3*i+2]   += -gk * dyk / dk
        grad[3*i+3]   +=  gk
        
        grad[3*j+1]   +=  gk * dxk / dk
        grad[3*j+2]   +=  gk * dyk / dk
        grad[3*j+3]   +=  gk

    # Gradient for -sum(r)
    grad[3::3] += -1.0

    # Gradient with respect to t (aspect ratio)
    # dw/dt = 2, dh/dt = -2
    # d/dt of softplus(xs + rs - w) = g2 * (-dw/dt) = -2*g2
    # d/dt of softplus(ys + rs - h) = g4 * (-dh/dt) = 2*g4
    grad[0] += -2.0 * np.sum(g2) + 2.0 * np.sum(g4)

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

def _delaunay_neighbors(centers):
    """Compute Delaunay neighbors for fast validity checks."""
    try:
        tri = Delaunay(centers)
        nbrs = [set() for _ in range(N)]
        for i, j, k in tri.simplices:
            nbrs[i].add(j); nbrs[i].add(k)
            nbrs[j].add(i); nbrs[j].add(k)
            nbrs[k].add(i); nbrs[k].add(j)
        return [sorted(s) for s in nbrs]
    except Exception:
        return [list(range(N)) for _ in range(N)]

def _fast_valid(circles, w, h, neighbors, tol=1e-6):
    """Fast validity check using Delaunay neighbors. Returns True if likely valid."""
    n = circles.shape[0]
    xs = circles[:, 0]; ys = circles[:, 1]; rs = circles[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[j] - tol
            if d2 < need * need:
                return False
    return True

def _hex_init(w, h, seed, aggressive=False):
    """Generate hex-packed initial configurations."""
    rng = random.Random(seed)
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
        [4, 4, 4, 4, 4, 5],
        [5, 4, 4, 4, 4],
        [3, 4, 4, 5, 5],
        [5, 5, 4, 4, 3],
        [6, 5, 5, 5],
        [5, 6, 5, 5],
        [5, 5, 6, 5],
        [5, 5, 5, 6],
    ]
    best = None
    best_val = -np.inf
    for row_counts in row_configs:
        if sum(row_counts) != N:
            continue
        n_rows = len(row_counts)
        max_cols = max(row_counts)
        
        if aggressive:
            # Inspiration style: larger initial radii
            est_r_h = w / (max_cols + 0.5)
            est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1.0)
        else:
            # Current style: more conservative
            est_r_h = w / (max_cols * 2.0 + 0.5)
            est_r_v = h / ((n_rows - 1) * np.sqrt(3) + 2.0)
            
        r_est = min(est_r_h, est_r_v) * 0.95
        if r_est < 0.01:
            continue
            
        circles = []
        for row_idx, n_cols in enumerate(row_counts):
            y = r_est + row_idx * r_est * np.sqrt(3)
            x_offset = r_est if row_idx % 2 == 1 else 0.0
            row_width = n_cols * 2 * r_est
            center_x = w / 2
            for col_idx in range(n_cols):
                x = center_x - row_width / 2 + x_offset + (col_idx + 0.5) * 2 * r_est
                circles.append([x, y, r_est])
        circles = np.array(circles)
        
        # Quick check: if aggressive, it might be invalid. We don't require validity for init,
        # but we prefer valid ones for the "best" selection within this function.
        if _strictly_valid(circles, w, h):
            val = float(np.sum(circles[:, 2]))
            if val > best_val:
                best_val = val
                best = circles.copy()
        else:
            # If no valid config found yet, keep the aggressive one as a fallback init
            if best is None:
                best = circles.copy()

    if best is None:
        # Fallback: grid
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

def _scheduled_optimize(x0, w, h, deadline, var_weight=0.0):
    """Run optimization with increasing beta."""
    phases = [
        (10.0, 50, 0.01),
        (50.0, 50, 0.01),
        (200.0, 50, 0.001),
        (500.0, 100, 0.0),
        (1000.0, 100, 0.0)
    ]
    
    current_x = x0
    for beta, max_iter, vw in phases:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_and_grad,
                current_x,
                args=(w, h, beta, vw),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            pass
    return current_x

def _scheduled_optimize_joint(x0, deadline):
    """Run joint optimization with increasing beta. x0 = [t, x1, y1, r1, ...]"""
    phases = [
        (10.0, 50),
        (50.0, 50),
        (200.0, 50),
        (500.0, 100),
        (1000.0, 100)
    ]
    
    current_x = x0
    for beta, max_iter in phases:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective_joint,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            pass
    return current_x

def _solve_lp_radii(circles, w, h):
    """
    Solve LP for optimal radii given fixed centers.
    Returns array of radii or None if infeasible.
    """
    xs = circles[:, 0]
    ys = circles[:, 1]
    
    c_obj = -np.ones(N)
    
    A_ub = []
    b_ub = []
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            row = np.zeros(N)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    for i in range(N):
        row = np.zeros(N)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xs[i])
        
        A_ub.append(row.copy())
        b_ub.append(w - xs[i])
        
        A_ub.append(row.copy())
        b_ub.append(ys[i])
        
        A_ub.append(row.copy())
        b_ub.append(h - ys[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(N)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT

    best_val = -1e10
    best_result = None
    best_w, best_h = 1.0, 1.0

    # Aspect ratios to try
    aspect_ratios = [
        1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5,
        0.85, 1.15, 0.75, 1.25, 0.65, 1.35, 0.55, 1.45, 0.95, 1.05,
        0.92, 1.08, 0.88, 1.12, 0.82, 1.18, 0.78, 1.22,
        0.72, 1.28, 0.68, 1.32, 0.62, 1.38, 0.58, 1.42, 0.52, 1.48
    ]

    # First pass: fixed aspect ratios
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break

        a = 1.0 / ar
        w = 2.0 / (1.0 + a)
        h = 2.0 * a / (1.0 + a)

        inits = []
        # Standard Hex
        inits.append(_hex_init(w, h, 0, aggressive=False))
        # Aggressive Hex (Inspiration style)
        inits.append(_hex_init(w, h, 1, aggressive=True))
        # Grid
        inits.append(_grid_init(w, h, 0))
        # Random
        inits.append(_random_init(w, h, 42))
        # Another Random
        inits.append(_random_init(w, h, 43))

        for init in inits:
            if time.time() > deadline - 10:
                break
            
            x0 = init.flatten()
            
            # Scheduled optimization
            x_opt = _scheduled_optimize(x0, w, h, deadline - 10)
            
            c = x_opt.reshape(-1, 3)
            # Clamp
            c[:, 0] = np.clip(c[:, 0], 0, w)
            c[:, 1] = np.clip(c[:, 1], 0, h)
            c[:, 2] = np.clip(c[:, 2], 1e-9, min(w, h)/2)
            
            # Fast check with Delaunay
            centers = c[:, :2]
            neighbors = _delaunay_neighbors(centers)
            
            if _fast_valid(c, w, h, neighbors, tol=1e-5):
                # Strict check
                if _strictly_valid(c, w, h):
                    val = float(np.sum(c[:, 2]))
                    if val > best_val:
                        best_val = val
                        best_result = c.copy()
                        best_w, best_h = w, h

    # Second pass: Joint aspect ratio optimization
    if time.time() < deadline - 15 and best_result is not None:
        # Use the best result found so far as a starting point for joint optimization
        # Convert to joint state: [t, x1, y1, r1, ...]
        t_init = best_w / 4.0  # w = 2t => t = w/2, but w = 2t so t = w/2? No, w = 2t, so t = w/2.
        # Wait, w = 2t, h = 2(1-t). Perimeter = 2(w+h) = 2(2t + 2(1-t)) = 4. Correct.
        # So t = w/2.
        t_init = best_w / 2.0
        t_init = np.clip(t_init, 0.15, 0.85)
        
        # Build joint state
        joint_state = np.zeros(1 + 3*N)
        joint_state[0] = t_init
        for i in range(N):
            joint_state[3*i+1] = best_result[i, 0]
            joint_state[3*i+2] = best_result[i, 1]
            joint_state[3*i+3] = best_result[i, 2]
        
        # Run joint optimization
        joint_opt = _scheduled_optimize_joint(joint_state, deadline - 15)
        
        t_opt = np.clip(joint_opt[0], 0.15, 0.85)
        w_opt = 2.0 * t_opt
        h_opt = 2.0 * (1.0 - t_opt)
        
        c_opt = np.zeros((N, 3))
        c_opt[:, 0] = joint_opt[1::3]
        c_opt[:, 1] = joint_opt[2::3]
        c_opt[:, 2] = joint_opt[3::3]
        
        # Clamp
        c_opt[:, 0] = np.clip(c_opt[:, 0], 0, w_opt)
        c_opt[:, 1] = np.clip(c_opt[:, 1], 0, h_opt)
        c_opt[:, 2] = np.clip(c_opt[:, 2], 1e-9, min(w_opt, h_opt)/2)
        
        # Fast check with Delaunay
        centers = c_opt[:, :2]
        neighbors = _delaunay_neighbors(centers)
        
        if _fast_valid(c_opt, w_opt, h_opt, neighbors, tol=1e-5):
            # Strict check
            if _strictly_valid(c_opt, w_opt, h_opt):
                val = float(np.sum(c_opt[:, 2]))
                if val > best_val:
                    best_val = val
                    best_result = c_opt.copy()
                    best_w, best_h = w_opt, h_opt

    if best_result is None:
        # Fallback
        w, h = 1.0, 1.0
        a = 5
        b = 5
        circles = []
        for i in range(N):
            col = i % a
            row = i // b
            r = 0.15
            x = (col + 0.5) * w / a
            y = (row + 0.5) * h / b
            circles.append([x, y, r])
        best_result = np.array(circles)
        best_val = float(np.sum(best_result[:, 2]))
        best_w, best_h = w, h

    # Final polish: LP radii solve
    if time.time() < deadline - 5:
        # Try to improve radii using LP
        lp_radii = _solve_lp_radii(best_result, best_w, best_h)
        if lp_radii is not None:
            c_lp = best_result.copy()
            c_lp[:, 2] = lp_radii
            if _strictly_valid(c_lp, best_w, best_h):
                val_lp = float(np.sum(c_lp[:, 2]))
                if val_lp > best_val:
                    best_val = val_lp
                    best_result = c_lp.copy()

    # Final polish: high-precision optimization
    if time.time() < deadline - 2:
        x_best = best_result.flatten()
        # Use a tighter schedule for polish
        x_polished = _scheduled_optimize(x_best, best_w, best_h, deadline - 2, var_weight=0.0)
        c_polished = x_polished.reshape(-1, 3)
        c_polished[:, 0] = np.clip(c_polished[:, 0], 0, best_w)
        c_polished[:, 1] = np.clip(c_polished[:, 1], 0, best_h)
        c_polished[:, 2] = np.clip(c_polished[:, 2], 1e-9, min(best_w, best_h)/2)
        
        if _strictly_valid(c_polished, best_w, best_h):
            val_polished = float(np.sum(c_polished[:, 2]))
            if val_polished > best_val:
                best_val = val_polished
                best_result = c_polished.copy()

    return best_result
