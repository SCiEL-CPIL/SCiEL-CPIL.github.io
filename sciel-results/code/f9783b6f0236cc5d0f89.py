"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Design: smooth penalty optimization with multi-start + adaptive SGD + L-BFGS-B polish.
Uses softplus smooth max for constraint violations, vectorized numpy operations.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 115.0
RNG = np.random.default_rng(20240517)

R_MIN = 1e-6
R_MAX = 0.5


def _is_valid(c, tol=1e-6):
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[mask] < s[mask] - tol):
        return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _softplus(v, T):
    a = v / T
    return T * np.where(a > 50, a, np.log1p(np.exp(a)))


def _smooth_max_0(v, T):
    return _softplus(v, T)


def _penalty_value(c, T, w=300.0):
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    wall = (_smooth_max_0(r - x, T) + _smooth_max_0(x + r - 1.0, T) +
            _smooth_max_0(r - y, T) + _smooth_max_0(y + r - 1.0, T))
    wall_pen = np.sum(wall)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    s = r[:, None] + r[None, :]
    overlap = _smooth_max_0(s - d, T)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(overlap[mask])
    return -np.sum(r) + w * (wall_pen + overlap_pen)


def _grad_penalty(c, T, w=300.0):
    """Compute gradient of penalty objective w.r.t. (x,y,r) for each circle."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    gx = np.zeros(N); gy = np.zeros(N); gr = np.zeros(N)
    
    # Wall gradients
    # d/dx of max(0, r-x)/T * T = sigmoid-like
    def _sp_grad(v, T):
        a = v / T
        return 1.0 / (1.0 + np.exp(-a))
    
    gw1 = _sp_grad(r - x, T)  # d/dx of sp(r-x) = -sp_grad
    gw2 = _sp_grad(x + r - 1.0, T)
    gw3 = _sp_grad(r - y, T)
    gw4 = _sp_grad(y + r - 1.0, T)
    
    gx += -gw1 * T + gw2 * T
    gy += -gw3 * T + gw4 * T
    gr += gw1 * T + gw2 * T + gw3 * T + gw4 * T
    
    # Overlap gradients (pairwise)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2) + 1e-15
    s = r[:, None] + r[None, :]
    v = s - d
    sg = _sp_grad(v, T) * T  # gradient of softplus wrt its argument, times T
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    sg_m = sg * mask
    
    # d/dx_i of sum over j of sp(s_ij - d_ij)
    # = sum_j sp_grad * (-dx_ij / d_ij)
    gx += np.sum(-sg_m * dx / d, axis=1)
    gy += np.sum(-sg_m * dy / d, axis=1)
    # d/dr_i: sum_j sp_grad (for all j != i)
    gr += np.sum(sg_m, axis=1) + np.sum(sg_m, axis=0)
    
    # Total: minimize -sum(r) + w*(wall + overlap)
    # d/dx: w * gx_wall + w * gx_overlap
    # d/dr: -1 + w * gr_wall + w * gr_overlap
    out = np.zeros((N, 3))
    out[:, 0] = w * gx
    out[:, 1] = w * gy
    out[:, 2] = -1.0 + w * gr
    return out


def _sgd_optimize(c0, T0=0.04, T_end=0.003, n_iter=4000, lr0=0.01, w=300.0):
    """Adaptive learning rate SGD on smooth penalty."""
    c = c0.copy()
    T = T0
    cool_rate = (T_end / T0) ** (1.0 / max(n_iter, 1))
    lr = lr0
    best_c = c.copy()
    best_val = _score(c)
    
    # Momentum
    v = np.zeros_like(c)
    beta = 0.9
    
    for it in range(n_iter):
        if time.time() > DEADLINE - 10.0:
            break
        
        T = T0 * (cool_rate ** it)
        lr = lr0 * (0.999 ** it)
        
        g = _grad_penalty(c, T, w)
        v = beta * v + (1 - beta) * g
        v_norm = np.sqrt(np.sum(v**2))
        if v_norm > 1e-10:
            v = v / v_norm * min(lr, 0.05)
        
        c_new = c - v
        c_new = _clamp(c_new)
        
        # Accept/reject with Metropolis
        cur_pen = _penalty_value(c, T, w)
        new_pen = _penalty_value(c_new, T, w)
        delta = new_pen - cur_pen
        if delta < 0 or RNG.random() < np.exp(-delta / max(T, 1e-8)):
            c = c_new
        
        if _is_valid(c):
            val = _score(c)
            if val > best_val:
                best_val = val
                best_c = c.copy()
    
    return best_c, best_val


def _lbfgs_polish(c0, T_start=0.015, w=200.0, max_iter=200):
    """L-BFGS-B with analytical gradient for final polish."""
    x0 = c0.flatten()
    best_c = c0.copy()
    best_val = _score(c0)
    
    bounds = []
    for _ in range(N):
        bounds.append((R_MIN, 1.0))
    for _ in range(N):
        bounds.append((R_MIN, 1.0))
    for _ in range(N):
        bounds.append((R_MIN, R_MAX))
    
    for stage in range(4):
        if time.time() > DEADLINE - 8.0:
            break
        T = T_start * (0.5 ** stage)
        
        def _obj(params):
            c = params.reshape(-1, 3)
            return _penalty_value(c, T, w)
        
        def _grad(params):
            c = params.reshape(-1, 3)
            g = _grad_penalty(c, T, w)
            return g.flatten()
        
        try:
            res = minimize(_obj, x0, jac=_grad, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            c_polished = res.x.reshape(-1, 3)
            c_polished = _clamp(c_polished)
            if _is_valid(c_polished):
                val = _score(c_polished)
                if val > best_val:
                    best_val = val
                    best_c = c_polished.copy()
            x0 = c_polished.flatten()
        except Exception:
            pass
    
    return best_c, best_val


def _coordinate_descent(c0, T=0.01, max_rounds=30, w=300.0):
    """Per-circle L-BFGS-B optimization."""
    c = c0.copy()
    best_c = c.copy()
    best_val = _score(c)
    
    bounds_i = [(R_MIN, 1.0), (R_MIN, 1.0), (R_MIN, R_MAX)]
    
    for rnd in range(max_rounds):
        if time.time() > DEADLINE - 10.0:
            break
        order = RNG.permutation(N)
        for i in order:
            if time.time() > DEADLINE - 8.0:
                break
            others = np.delete(c, i, axis=0)
            
            def _obj_i(params):
                x_i, y_i, r_i = params
                r_i = np.clip(r_i, R_MIN, R_MAX)
                x_i = np.clip(x_i, r_i, 1.0 - r_i)
                y_i = np.clip(y_i, r_i, 1.0 - r_i)
                
                wall = (_smooth_max_0(r_i - x_i, T) + _smooth_max_0(x_i + r_i - 1.0, T) +
                        _smooth_max_0(r_i - y_i, T) + _smooth_max_0(y_i + r_i - 1.0, T))
                wall_pen = np.sum(wall)
                
                dx = x_i - others[:, 0]
                dy = y_i - others[:, 1]
                d = np.sqrt(dx**2 + dy**2) + 1e-15
                s = r_i + others[:, 2]
                ov = _smooth_max_0(s - d, T)
                overlap_pen = np.sum(ov)
                
                return -r_i + w * (wall_pen + overlap_pen)
            
            try:
                x0_i = c[i, :].copy()
                res = minimize(_obj_i, x0_i, method='L-BFGS-B',
                              bounds=bounds_i, options={'maxiter': 40, 'ftol': 1e-10})
                new_i = np.clip(res.x, [R_MIN, R_MIN, R_MIN], [1.0, 1.0, R_MAX])
                new_i[0] = np.clip(new_i[0], new_i[2], 1.0 - new_i[2])
                new_i[1] = np.clip(new_i[1], new_i[2], 1.0 - new_i[2])
                c[i] = new_i
            except Exception:
                pass
        
        c = _clamp(c)
        val = _score(c)
        if val > best_val:
            best_val = val
            best_c = c.copy()
        T *= 0.9
        if T < 0.003:
            break
    
    return best_c, best_val


def _seed_hex():
    positions = []
    r = 0.16
    dx = 2 * r
    dy = r * np.sqrt(3)
    for row in range(5):
        for col in range(5):
            x = r + col * dx + (row % 2) * dx / 2
            y = r + row * dy
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                positions.append((x, y, r))
    while len(positions) < N:
        x = RNG.uniform(0.08, 0.92)
        y = RNG.uniform(0.08, 0.92)
        positions.append((x, y, 0.08))
    return np.array(positions[:N])


def _seed_corner():
    positions = []
    for cx, cy, cr in [(0.12, 0.12, 0.12), (0.88, 0.12, 0.12),
                       (0.12, 0.88, 0.12), (0.88, 0.88, 0.12),
                       (0.5, 0.08, 0.08), (0.5, 0.92, 0.08),
                       (0.08, 0.5, 0.08), (0.92, 0.5, 0.08),
                       (0.5, 0.5, 0.10)]:
        positions.append((cx, cy, cr))
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        positions.append((x, y, RNG.uniform(0.04, 0.09)))
    return np.array(positions[:N])


def _seed_random():
    x = RNG.uniform(0.1, 0.9, N)
    y = RNG.uniform(0.1, 0.9, N)
    r = RNG.uniform(0.04, 0.12, N)
    return np.column_stack([x, y, r])


def _seed_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cr = 0.5 / side - 0.01
            cells.append((cx, cy, cr))
    return np.array(cells[:N])


def construct():
    t0 = time.time()
    
    initializers = [_seed_hex, _seed_corner, _seed_random, _seed_grid]
    best_c = None
    best_val = -np.inf
    
    # Phase 1: SGD + Metropolis from multiple starts
    for init_func in initializers:
        if time.time() > DEADLINE - 55.0:
            break
        c0 = init_func()
        cc, val = _sgd_optimize(c0, T0=0.05, T_end=0.003, n_iter=3000, lr0=0.012, w=300.0)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 2: Additional random starts
    for i in range(6):
        if time.time() > DEADLINE - 35.0:
            break
        c0 = _seed_random()
        cc, val = _sgd_optimize(c0, T0=0.04, T_end=0.002, n_iter=2000, lr0=0.01, w=300.0)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 3: Coordinate descent refinement
    if best_c is not None and time.time() < DEADLINE - 20.0:
        cc, val = _coordinate_descent(best_c, T=0.012, max_rounds=25, w=300.0)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 4: L-BFGS-B global polish
    if best_c is not None and time.time() < DEADLINE - 10.0:
        cc, val = _lbfgs_polish(best_c, T_start=0.012, w=200.0, max_iter=150)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Phase 5: Final coordinate descent
    if best_c is not None and time.time() < DEADLINE - 6.0:
        cc, val = _coordinate_descent(best_c, T=0.008, max_rounds=20, w=300.0)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Final L-BFGS-B
    if best_c is not None and time.time() < DEADLINE - 4.0:
        cc, val = _lbfgs_polish(best_c, T_start=0.008, w=150.0, max_iter=100)
        if val > best_val:
            best_val = val
            best_c = cc.copy()
    
    # Safety fallback
    if best_c is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_c = np.array(cells[:N])
    
    return best_c
