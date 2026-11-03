"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Combines hexagonal structured initialization with softplus gradient optimization.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 145.0  # Safety margin

def _softplus(x, beta=100.0):
    """Stable softplus function."""
    xb = beta * x
    return np.where(xb > 0,
                    x + np.log1p(np.exp(-xb)) / beta,
                    np.log1p(np.exp(xb)) / beta)

def _softplus_grad(x, beta=100.0):
    """Gradient of softplus (sigmoid scaled by beta)."""
    xb = beta * x
    return np.where(xb > 0,
                    1.0 / (1.0 + np.exp(-xb)),
                    np.exp(xb) / (1.0 + np.exp(xb)))

def _objective_and_grad(x, w, h, beta=100.0):
    """
    Objective: -sum(r) + penalties for boundary violations and overlaps.
    x: array of shape (3*N,) = [x1, y1, r1, x2, y2, r2, ...]
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

    # Pairwise overlap penalties - vectorized
    for i in range(N):
        dx = xs[i] - xs[i + 1:]
        dy = ys[i] - ys[i + 1:]
        d = np.sqrt(dx * dx + dy * dy + 1e-12)
        v = rs[i] + rs[i + 1:] - d
        sp = _softplus(v, beta)
        g = _softplus_grad(v, beta)
        
        obj += np.sum(sp)
        
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
    """Check if circles are within bounds and disjoint."""
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
    # Use Delaunay for O(n) neighbor check instead of O(n^2)
    try:
        from scipy.spatial import Delaunay
        pts = np.column_stack([xs, ys])
        tri = Delaunay(pts)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    dx = xs[a] - xs[b]
                    dy = ys[a] - ys[b]
                    d2 = dx * dx + dy * dy
                    need = rs[a] + rs[b] - tol
                    if d2 < need * need:
                        return False
        # Also check non-neighbor pairs that might be close
        # (Delaunay only guarantees neighbors; add a few extra checks)
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            # Only check pairs with distance < 3*max_r to limit work
            max_r = np.max(rs)
            mask = d2 < (3 * max_r) ** 2
            if np.any(mask & (d2 < need[need > 0] ** 2)):
                # Verify more carefully
                for k in np.where(mask)[0]:
                    j = i + 1 + k
                    if d2[k] < (rs[i] + rs[j] - tol) ** 2:
                        return False
    except Exception:
        # Fallback to O(n^2)
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            d2 = dx * dx + dy * dy
            need = rs[i] + rs[i + 1:] - tol
            if np.any(d2 < need * need):
                return False
    return True

def _hex_init(w, h, seed=0):
    """
    Generate hex-packed initial configurations with various row layouts.
    """
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
        [4, 4, 4, 5, 4],
        [3, 4, 4, 4, 6],
        [6, 4, 4, 4, 3],
    ]
    
    best = None
    best_val = -np.inf
    
    # Shuffle configs for diversity across seeds
    shuffled = row_configs.copy()
    rng.shuffle(shuffled)
    
    for row_counts in shuffled:
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

def _grid_init(w, h):
    """Simple grid initialization."""
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
    """Random initialization."""
    rng = np.random.RandomState(seed)
    r0 = min(w, h) / 2.0 * 0.2
    circles = np.zeros((N, 3))
    circles[:, 0] = rng.uniform(r0, w - r0, N)
    circles[:, 1] = rng.uniform(r0, h - r0, N)
    circles[:, 2] = r0 * rng.uniform(0.8, 1.2, N)
    return circles

def _scale_to_perim(circles, target_perim=4.0):
    """Scale circles to fit within a bounding box of target perimeter."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    width = max_x - min_x
    height = max_y - min_y
    perim = 2.0 * (width + height)
    
    if perim <= target_perim + 1e-9:
        return circles
    
    scale = target_perim / perim
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    
    xs_new = cx + (xs - cx) * scale
    ys_new = cy + (ys - cy) * scale
    rs_new = rs * scale
    
    return np.column_stack([xs_new, ys_new, rs_new])

def _polish(circles, w, h, deadline):
    """Multi-phase polish with increasing beta for tighter convergence."""
    current = circles.copy()
    
    # Phase 1: moderate beta, more iterations
    beta1 = 50.0
    x0 = current.flatten()
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-6, 1.0))
    
    try:
        res1 = minimize(
            lambda p: _objective_and_grad(p, w, h, beta1)[0],
            x0,
            jac=lambda p: _objective_and_grad(p, w, h, beta1)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        if res1.success or res1.fun < 1e6:
            c1 = res1.x.reshape(-1, 3)
            if _strictly_valid(c1, w, h):
                current = c1
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return current
    
    # Phase 2: high beta, fine-tuning
    beta2 = 200.0
    x0 = current.flatten()
    try:
        res2 = minimize(
            lambda p: _objective_and_grad(p, w, h, beta2)[0],
            x0,
            jac=lambda p: _objective_and_grad(p, w, h, beta2)[1],
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-16}
        )
        if res2.success or res2.fun < 1e6:
            c2 = res2.x.reshape(-1, 3)
            if _strictly_valid(c2, w, h):
                current = c2
    except Exception:
        pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_value = -np.inf
    best_result = None
    
    # Focus on fewer, better aspect ratios with more optimization depth
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.85, 1.15, 0.75, 1.25]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Generate diverse initializations
        inits = []
        
        # Hex inits with different seeds for diversity
        for seed in range(3):
            circ = _hex_init(w, h, seed)
            inits.append(circ)
            
        # Grid init
        circ = _grid_init(w, h)
        inits.append(circ)
        
        # Random inits
        for seed in [1, 2]:
            circ = _random_init(w, h, seed)
            inits.append(circ)
            
        for init_circles in inits:
            if time.time() > deadline - 8:
                break
                
            x0 = init_circles.flatten()
            
            bounds = []
            for i in range(N):
                bounds.append((0, w))
                bounds.append((0, h))
                bounds.append((1e-6, 1.0))
                
            try:
                # Initial optimization with moderate beta
                res = minimize(
                    lambda p: _objective_and_grad(p, w, h, 50.0)[0],
                    x0,
                    jac=lambda p: _objective_and_grad(p, w, h, 50.0)[1],
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12}
                )
                
                if res.success or res.fun < 1e6:
                    x_opt = res.x
                    circles_opt = x_opt.reshape(-1, 3)
                    
                    if _strictly_valid(circles_opt, w, h):
                        # Apply multi-phase polish
                        polished = _polish(circles_opt, w, h, deadline)
                        val = np.sum(polished[:, 2])
                        if val > best_value:
                            best_value = val
                            best_result = polished.copy()
                            
            except Exception:
                pass
                
    # Final scaling to ensure perimeter <= 4
    if best_result is not None:
        best_result = _scale_to_perim(best_result, 4.0)
    else:
        best_result = _grid_init(1.0, 1.0)
        best_result = _scale_to_perim(best_result, 4.0)
        
    return best_result
