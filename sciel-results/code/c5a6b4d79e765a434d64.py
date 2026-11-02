"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _validate_and_score(circles):
    """Check validity and compute score. circles: (N,3) array [x,y,r]."""
    if circles.shape != (N, 3):
        return None
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return None
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9) or np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return None
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
        return None
    return float(np.sum(r))


def _grid_initial(n=N):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((cx, cy, r))
    return np.array(cells[:n])


def _hex_initial(n=N, seed=0):
    """Hexagonal structured initialization."""
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    cols = 5
    rows = 6
    dx = 1.0 / (cols + 0.5)
    dy = 1.0 / (rows + 0.5)
    r_base = min(dx, dy) * 0.48
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x_offset = 0.5 * dx if row % 2 == 1 else 0
            cx = (col + 0.5 + x_offset) * dx
            cy = (row + 0.5) * dy
            cx += 0.005 * rng.randn()
            cy += 0.005 * rng.randn()
            cx = np.clip(cx, r_base, 1 - r_base)
            cy = np.clip(cy, r_base, 1 - r_base)
            circles[idx, 0] = cx
            circles[idx, 1] = cy
            circles[idx, 2] = r_base
            idx += 1
    return circles


def _random_initial(n=N, seed=42):
    rng = np.random.RandomState(seed)
    circles = np.zeros((n, 3))
    circles[:, 2] = 0.015 + 0.015 * rng.rand(n)
    circles[:, 0] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    circles[:, 1] = circles[:, 2] + (1 - 2 * circles[:, 2]) * rng.rand(n)
    return circles


def _penalty_objective(params, n=N, pw=500.0):
    """Objective to minimize: -sum(r) + penalty for violations. Vectorized."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    obj = -np.sum(r)
    # Boundary violations
    bv = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
         np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    obj += pw * np.sum(bv)
    # Overlap penalties
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap = np.maximum(0, r[i_idx] + r[j_idx] - dist)
    obj += pw * np.sum(overlap)
    return obj


def _lbfgs_optimize(circles, time_limit=20, seed=0):
    """L-BFGS-B optimization with penalty method. Vectorized objective."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    params = np.concatenate([x, y, r])
    
    # Bounds: positions in [0,1], radii in [0.001, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.001, 0.5)] * n
    
    start_time = time.time()
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_params = params.copy()
    if best_score is None:
        best_score = 0.0
    
    # Callback to track best
    def callback(params_cb):
        nonlocal best_score, best_params
        cx = params_cb[:n]
        cy = params_cb[n:2*n]
        cr = params_cb[2*n:3*n]
        # Quick validity check
        if np.any(cr <= 0) or np.any(cx - cr < -1e-7) or np.any(cx + cr > 1 + 1e-7) or \
           np.any(cy - cr < -1e-7) or np.any(cy + cr > 1 + 1e-7):
            return False
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = cx[i_idx] - cx[j_idx]
        dy = cy[i_idx] - cy[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < cr[i_idx] + cr[j_idx] - 1e-7):
            return False
        score = float(np.sum(cr))
        if score > best_score:
            best_score = score
            best_params = params_cb.copy()
        # Time check
        if time.time() - start_time > time_limit - 2:
            return True
        return False
    
    try:
        res = minimize(
            _penalty_objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 100000},
            callback=callback
        )
    except Exception:
        pass
    
    # Final check
    final_params = best_params
    fx = final_params[:n]
    fy = final_params[n:2*n]
    fr = final_params[2*n:3*n]
    final_circles = np.column_stack([fx, fy, fr])
    final_score = _validate_and_score(final_circles)
    if final_score is not None and final_score > best_score:
        best_score = final_score
    
    return final_circles, best_score


def _fast_local_refine(circles, time_limit=15, seed=0):
    """Fast vectorized local refinement: position separation + radius growth."""
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Phase 1: Separate overlapping circles (position only)
    pos_time = time_limit * 0.4
    lr = 0.01
    it = 0
    while time.time() - start_time < pos_time:
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[i_idx] + r[j_idx] - dist
        active = overlap > 0
        if not np.any(active):
            break
        repulsion = np.where(active, 50.0 * overlap / dist, 0.0)
        gx = np.zeros(n)
        gy = np.zeros(n)
        np.add.at(gx, i_idx, -repulsion * dx / dist)
        np.add.at(gy, i_idx, -repulsion * dy / dist)
        np.add.at(gx, j_idx, repulsion * dx / dist)
        np.add.at(gy, j_idx, repulsion * dy / dist)
        # Boundary repulsion
        gx += np.where(x < r, 200.0, 0.0) - np.where(1 - x < r, 200.0, 0.0)
        gy += np.where(y < r, 200.0, 0.0) - np.where(1 - y < r, 200.0, 0.0)
        x += lr * gx
        y += lr * gy
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        lr *= 0.999
        it += 1
    
    # Phase 2: Grow radii
    grow_time = time_limit * 0.6
    grow_lr = 0.002
    it2 = 0
    while time.time() - start_time < grow_time:
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        dist = np.maximum(dist, 1e-12)
        overlap = r[i_idx] + r[j_idx] - dist
        gr = np.ones(n)
        # Boundary constraint
        gr -= np.where(x < r, 300.0, 0.0)
        gr -= np.where(1 - x < r, 300.0, 0.0)
        gr -= np.where(y < r, 300.0, 0.0)
        gr -= np.where(1 - y < r, 300.0, 0.0)
        # Overlap penalty
        pen = np.where(overlap > 0, 100.0, 0.0)
        pen_i = np.zeros(n)
        pen_j = np.zeros(n)
        np.add.at(pen_i, i_idx, pen)
        np.add.at(pen_j, j_idx, pen)
        gr -= pen_i + pen_j
        r += grow_lr * gr
        r = np.maximum(r, 0.001)
        max_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, np.maximum(max_r, 0.001))
        grow_lr *= 0.9999
        it2 += 1
        if it2 % 100 == 0:
            cur = np.column_stack([x, y, r])
            sc = _validate_and_score(cur)
            if sc is not None and sc > best_score:
                best_score = sc
                best_circles = cur.copy()
    
    # Final check
    cur = np.column_stack([x, y, r])
    sc = _validate_and_score(cur)
    if sc is not None and sc > best_score:
        best_score = sc
        best_circles = cur.copy()
    
    return best_circles, best_score


def _anneal_refine(circles, time_limit=25, seed=0):
    """Simulated annealing with vectorized scoring."""
    rng = np.random.RandomState(seed)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    n = len(r)
    start_time = time.time()
    
    best_score = _validate_and_score(np.column_stack([x, y, r]))
    best_circles = np.column_stack([x, y, r]).copy()
    if best_score is None:
        best_score = 0.0
    current_score = best_score
    
    t = 0.03
    n_iter = 100000
    cooling = (0.0005 / t) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randint(n)
        move_type = rng.rand()
        new_x = x.copy()
        new_y = y.copy()
        new_r = r.copy()
        
        if move_type < 0.4:
            delta = 0.015 * rng.randn(2)
            new_x[i] += delta[0]
            new_y[i] += delta[1]
        elif move_type < 0.7:
            new_r[i] *= (1 + 0.08 * rng.randn())
            new_r[i] = max(0.005, min(new_r[i], 0.5))
        else:
            j = rng.randint(n)
            if j == i:
                j = (i + 1) % n
            delta_r = 0.008 * rng.randn()
            new_r[i] = max(0.005, new_r[i] + delta_r)
            new_r[j] = max(0.005, new_r[j] - delta_r)
        
        sc = _validate_and_score(np.column_stack([new_x, new_y, new_r]))
        if sc is not None:
            if sc > current_score:
                x, y, r = new_x, new_y, new_r
                current_score = sc
                if sc > best_score:
                    best_score = sc
                    best_circles = np.column_stack([x, y, r]).copy()
            else:
                delta = sc - current_score
                if delta < 0:
                    prob = np.exp(delta / max(t, 1e-10))
                    if rng.rand() < prob:
                        x, y, r = new_x, new_y, new_r
                        current_score = sc
        t *= cooling
    
    return best_circles, best_score


def construct():
    """Construct optimal circle packing in unit square for N=26 circles."""
    start_time = time.time()
    time_budget = 130  # safety margin from 150s kill
    safety_margin = 5
    
    best_circles = None
    best_score = 0.0
    
    # Generate diverse initializations
    initializations = [
        _grid_initial(N),
        _hex_initial(N, seed=0),
        _hex_initial(N, seed=1),
        _random_initial(N, seed=42),
        _random_initial(N, seed=123),
    ]
    
    # For each initialization: L-BFGS-B optimize, then local refine
    for idx, init in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed - safety_margin
        if remaining < 15:
            break
        
        # L-BFGS-B optimization
        lbfgs_time = min(remaining * 0.5, 25)
        circles, score = _lbfgs_optimize(init, time_limit=lbfgs_time, seed=idx)
        
        if score > best_score:
            best_score = score
            best_circles = circles.copy()
        
        # Local refinement
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed - safety_margin
        if remaining < 10:
            break
        local_time = min(remaining * 0.4, 15)
        refined, r_score = _fast_local_refine(circles, time_limit=local_time, seed=idx)
        if r_score > best_score:
            best_score = r_score
            best_circles = refined.copy()
    
    # Annealing refinement on best result
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed - safety_margin
    if remaining > 20 and best_circles is not None:
        anneal_time = min(remaining * 0.6, 30)
        annealed, a_score = _anneal_refine(best_circles, time_limit=anneal_time, seed=99)
        if a_score > best_score:
            best_score = a_score
            best_circles = annealed.copy()
    
    # Final L-BFGS-B polish on best result
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed - safety_margin
    if remaining > 15 and best_circles is not None:
        polish_time = min(remaining * 0.5, 20)
        polished, p_score = _lbfgs_optimize(best_circles, time_limit=polish_time, seed=200)
        if p_score > best_score:
            best_score = p_score
            best_circles = polished.copy()
    
    # Final local refinement
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed - safety_margin
    if remaining > 10 and best_circles is not None:
        final_time = min(remaining * 0.5, 15)
        final_refined, f_score = _fast_local_refine(best_circles, time_limit=final_time, seed=300)
        if f_score > best_score:
            best_score = f_score
            best_circles = final_refined.copy()
    
    # Final validation
    if best_circles is None:
        return _grid_initial(N)
    
    final_score = _validate_and_score(best_circles)
    if final_score is None:
        return _grid_initial(N)
    
    return best_circles

# EVOLVE-BLOCK-END
