"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 21
TIME_LIMIT = 145.0

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

    # 2. Pairwise Non-overlap Constraints
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

def _hex_init(w, h, seed):
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
        [4, 4, 4, 4, 5],
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
    xs = rng.uniform(0.1*w, 0.9*w, N)
    ys = rng.uniform(0.1*h, 0.9*h, N)
    rs = np.full(N, 0.08 * min(w, h))
    return np.column_stack([xs, ys, rs])

def _scale_to_perim(circles, target_perim=4.0):
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    perim = 2.0 * ((max_x - min_x) + (max_y - min_y))
    if perim <= target_perim + 1e-9:
        return circles, perim
    scale = target_perim / perim
    cx = np.mean(xs)
    cy = np.mean(ys)
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    return np.column_stack([xs_new, ys_new, rs_new]), target_perim

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    random.seed(42)
    np.random.seed(42)
    
    best_value = -np.inf
    best_result = None
    
    aspect_ratios = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Generate initializations
        inits = []
        
        # Hex
        hex_circ = _hex_init(w, h, 42)
        inits.append(hex_circ)
        
        # Grid
        grid_circ = _grid_init(w, h, 42)
        inits.append(grid_circ)
        
        # Random
        for seed in [1, 2, 3, 42, 100]:
            rand_circ = _random_init(w, h, seed)
            inits.append(rand_circ)
        
        for init_circ in inits:
            if time.time() > deadline - 10:
                break
            
            # Try different beta values and var_weight
            for beta in [50.0, 100.0, 200.0]:
                if time.time() > deadline - 8:
                    break
                
                for var_weight in [0.0, 0.01, 0.1]:
                    if time.time() > deadline - 5:
                        break
                    
                    try:
                        # SLSQP
                        res = minimize(
                            lambda p, b=beta, vw=var_weight: _objective_and_grad(p, w, h, b, vw),
                            init_circ.flatten(),
                            jac=True,
                            method='SLSQP',
                            bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                            options={'maxiter': 300, 'ftol': 1e-12}
                        )
                        p_opt = res.x.reshape(-1, 3)
                        
                        if _strictly_valid(p_opt, w, h):
                            val = np.sum(p_opt[:, 2])
                            if val > best_value:
                                best_value = val
                                best_result = p_opt.copy()
                    except Exception:
                        pass
                    
                    if time.time() > deadline - 5:
                        break
                    
                    try:
                        # L-BFGS-B
                        res2 = minimize(
                            lambda p, b=beta, vw=var_weight: _objective_and_grad(p, w, h, b, vw),
                            init_circ.flatten(),
                            jac=True,
                            method='L-BFGS-B',
                            bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                            options={'maxiter': 200, 'ftol': 1e-12}
                        )
                        p_opt2 = res2.x.reshape(-1, 3)
                        
                        if _strictly_valid(p_opt2, w, h):
                            val2 = np.sum(p_opt2[:, 2])
                            if val2 > best_value:
                                best_value = val2
                                best_result = p_opt2.copy()
                    except Exception:
                        pass
        
        # Polish best result found so far for this aspect ratio
        if best_result is not None and time.time() < deadline - 5:
            try:
                res_polish = minimize(
                    lambda p: _objective_and_grad(p, w, h, 500.0, 0.0),
                    best_result.flatten(),
                    jac=True,
                    method='SLSQP',
                    bounds=[(0, w)] * N + [(0, h)] * N + [(1e-4, 0.5)] * N,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                p_polish = res_polish.x.reshape(-1, 3)
                if _strictly_valid(p_polish, w, h):
                    val_polish = np.sum(p_polish[:, 2])
                    if val_polish > best_value:
                        best_value = val_polish
                        best_result = p_polish.copy()
            except Exception:
                pass
    
    # Final refinement: local search with perturbation
    if best_result is not None and time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = best_result + np.random.randn(*best_result.shape) * 0.002
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 2.0)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 2.0)
            perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-4, 0.5)
            
            # Quick validity check
            if _strictly_valid(perturbed, 1.0, 1.0):
                # Re-optimize from perturbed
                try:
                    res_pert = minimize(
                        lambda p: _objective_and_grad(p, 1.0, 1.0, 100.0, 0.0),
                        perturbed.flatten(),
                        jac=True,
                        method='SLSQP',
                        bounds=[(0, 1.0)] * N + [(0, 1.0)] * N + [(1e-4, 0.5)] * N,
                        options={'maxiter': 100, 'ftol': 1e-12}
                    )
                    p_pert = res_pert.x.reshape(-1, 3)
                    if _strictly_valid(p_pert, 1.0, 1.0):
                        val_pert = np.sum(p_pert[:, 2])
                        if val_pert > best_value:
                            best_value = val_pert
                            best_result = p_pert.copy()
                except Exception:
                    pass
    
    # Return result
    if best_result is None:
        # Fallback: simple grid
        xs, ys = _grid_init(1.0, 1.0, 42)[:, :2].T
        r = 0.08
        best_result = np.column_stack([xs, ys, np.full(N, r)])
    
    # Scale to ensure perimeter <= 4
    best_result, _ = _scale_to_perim(best_result, 4.0)
    
    return best_result
