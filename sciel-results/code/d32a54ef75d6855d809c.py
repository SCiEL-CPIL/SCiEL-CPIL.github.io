import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin before 150s kill

def _pairwise_distances(x, y):
    """Compute pairwise distances between N points (vectorized)."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx**2 + dy**2)

def _is_valid(x, y, r, w, h, tol=1e-7):
    """Check if packing is valid."""
    if np.any(r < -tol):
        return False
    if np.any(x < r - tol) or np.any(x > w - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > h - r + tol):
        return False
    dist_mat = _pairwise_distances(x, y)
    for i in range(N):
        for j in range(i+1, N):
            if dist_mat[i, j] < r[i] + r[j] - tol:
                return False
    return True

def _slsqp_objective(vars_):
    r = vars_[2::3]
    return -np.sum(r)

def _slsqp_cons_inside(vars_, w, h):
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    return np.concatenate([
        x - r,
        w - x - r,
        y - r,
        h - y - r
    ])

def _slsqp_cons_non_overlap(vars_):
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    cons = []
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - r[i] - r[j])
    return np.array(cons)

def _slsqp_cons_positive_r(vars_):
    r = vars_[2::3]
    return r

def _slsqp_polish(x0, w, h, maxiter=200):
    """Run SLSQP polish on a feasible starting point."""
    constraints = [
        {'type': 'ineq', 'fun': lambda v: _slsqp_cons_inside(v, w, h)},
        {'type': 'ineq', 'fun': _slsqp_cons_non_overlap},
        {'type': 'ineq', 'fun': _slsqp_cons_positive_r}
    ]
    
    bounds = [(0, w)] * N + [(0, h)] * N + [(0, 1.0)] * N
    
    result = minimize(
        _slsqp_objective, x0, method='SLSQP',
        constraints=constraints, bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-10}
    )
    return result.x

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_wh = (1.0, 1.0)
    
    np.random.seed(42)
    
    aspect_ratios = [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
    
    # Phase 1: Penalty-based search
    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 25:
            break
        
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        if h <= 0.1:
            continue
        
        time_remaining = DEADLINE - 25 - elapsed
        max_restarts = max(2, min(6, int(time_remaining / 4.0)))
        
        for restart in range(max_restarts):
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 20:
                break
            
            # Grid initialization
            cols = int(np.sqrt(N * w / h))
            if cols < 2: cols = 2
            rows = int(np.ceil(N / cols))
            while cols * rows < N:
                if cols <= rows:
                    cols += 1
                else:
                    rows += 1
            
            positions = np.zeros((N, 2))
            idx = 0
            for r_idx in range(rows):
                for c_idx in range(cols):
                    if idx >= N:
                        break
                    x = (c_idx + 0.5) * w / cols
                    y = (r_idx + 0.5) * h / rows
                    
                    if restart > 0:
                        jitter = 0.15 * min(w/cols, h/rows)
                        x += np.random.uniform(-jitter, jitter)
                        y += np.random.uniform(-jitter, jitter)
                    
                    x = np.clip(x, 0.02, w - 0.02)
                    y = np.clip(y, 0.02, h - 0.02)
                    
                    positions[idx] = [x, y]
                    idx += 1
            
            # Estimate initial radius
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                    min_dist = min(min_dist, d)
            for i in range(N):
                min_dist = min(min_dist, positions[i,0], w-positions[i,0], positions[i,1], h-positions[i,1])
            
            r0 = 0.5 * min_dist
            radii = np.full(N, r0)
            
            x0 = np.zeros(3*N)
            for i in range(N):
                x0[3*i] = positions[i, 0]
                x0[3*i+1] = positions[i, 1]
                x0[3*i+2] = radii[i]
            
            # Try SLSQP directly (it's been working well)
            try:
                constraints = [
                    {'type': 'ineq', 'fun': lambda v, ww=w, hh=h: _slsqp_cons_inside(v, ww, hh)},
                    {'type': 'ineq', 'fun': _slsqp_cons_non_overlap},
                    {'type': 'ineq', 'fun': _slsqp_cons_positive_r}
                ]
                bounds = [(0, w)] * N + [(0, h)] * N + [(0, 1.0)] * N
                
                result = minimize(
                    _slsqp_objective, x0, method='SLSQP',
                    constraints=constraints, bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-8}
                )
                
                x_res = result.x[0::3]
                y_res = result.x[1::3]
                r_res = result.x[2::3]
                
                if _is_valid(x_res, y_res, r_res, w, h, tol=1e-6):
                    score = np.sum(r_res)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_res, y_res, r_res)).copy()
                        best_wh = (w, h)
                else:
                    # Slight repair
                    r_rep = r_res * 0.998
                    x_rep = np.clip(x_res, r_rep, w - r_rep)
                    y_rep = np.clip(y_res, r_rep, h - r_rep)
                    if _is_valid(x_rep, y_rep, r_rep, w, h, tol=1e-6):
                        score = np.sum(r_rep)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_rep, y_rep, r_rep)).copy()
                            best_wh = (w, h)
                
            except Exception:
                continue
    
    # Phase 2: Polish the best result
    if best_result is not None and best_score > 0:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 3:
            w, h = best_wh
            x0 = best_result.ravel().copy()
            try:
                polished = _slsqp_polish(x0, w, h, maxiter=300)
                x_pol = polished[0::3]
                y_pol = polished[1::3]
                r_pol = polished[2::3]
                
                if _is_valid(x_pol, y_pol, r_pol, w, h, tol=1e-7):
                    score = np.sum(r_pol)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack((x_pol, y_pol, r_pol)).copy()
                else:
                    # Try with slightly shrunk radii
                    r_shrunk = r_pol * 0.999
                    x_shrunk = np.clip(x_pol, r_shrunk, w - r_shrunk)
                    y_shrunk = np.clip(y_pol, r_shrunk, h - r_shrunk)
                    if _is_valid(x_shrunk, y_shrunk, r_shrunk, w, h, tol=1e-7):
                        score = np.sum(r_shrunk)
                        if score > best_score:
                            best_score = score
                            best_result = np.column_stack((x_shrunk, y_shrunk, r_shrunk)).copy()
            except Exception:
                pass
    
    # Fallback
    if best_result is None or best_score < 0:
        w = 1.0
        h = 1.0
        r = 0.09
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * w / 5
                y = (j + 0.5) * h / 5
                positions.append([x, y])
                radii.append(r)
                count += 1
        
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)
    
    return best_result
