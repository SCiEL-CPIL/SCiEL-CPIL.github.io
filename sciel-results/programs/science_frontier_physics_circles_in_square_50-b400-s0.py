"""Packing of 50 equal circles in a unit square via SLSQP refinement + Binary Search + Soft-Min Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
N_PAIRS = len(IU[0])

# Global deadline guard
_DEADLINE = 0.0

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _soft_min_obj(x, beta):
    """Objective: minimize -soft_min of all constraint distances (maximize the min)."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    m = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - m))
    log_sum = np.log(np.sum(exp_terms))
    r_soft = m - log_sum / beta
    return -r_soft

def _soft_min_grad(x, beta):
    """Gradient of -soft_min_r."""
    p = x.reshape(N, 2)
    d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                        np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
    diff = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_d = dist[IU] / 2.0
    
    all_d = np.concatenate([d_wall, pair_d])
    m = np.max(all_d)
    exp_terms = np.exp(-beta * (all_d - m))
    weights = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros((N, 2))
    
    # Wall gradients
    d_left = p[:, 0] + HALF
    d_right = HALF - p[:, 0]
    d_bottom = p[:, 1] + HALF
    d_top = HALF - p[:, 1]
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    min_idx = np.argmin(walls, axis=1)
    
    for i in range(N):
        w = weights[i]
        if w > 1e-12:
            if min_idx[i] == 0:
                grad[i, 0] += w
            elif min_idx[i] == 1:
                grad[i, 0] -= w
            elif min_idx[i] == 2:
                grad[i, 1] += w
            else:
                grad[i, 1] -= w
    
    # Pair gradients
    for k in range(N_PAIRS):
        i, j = IU[0][k], IU[1][k]
        w = weights[N + k]
        if w > 1e-12:
            dij = p[i] - p[j]
            dist_ij = np.linalg.norm(dij)
            if dist_ij > 1e-12:
                g = w * dij / (2.0 * dist_ij)
                grad[i] += g
                grad[j] -= g
    
    return -grad.ravel()

def _slsqp_max_radius(x0, deadline, maxiter=500):
    """
    Maximize radius r subject to constraints:
    1. d_ij >= 2r
    2. d_wall >= r
    Variables: x (2N coords) + r (1 scalar)
    """
    def obj(v):
        # Minimize -r
        return -v[-1]
    
    def jac(v):
        g = np.zeros_like(v)
        g[-1] = -1.0
        return g
    
    # Constraints: h(v) >= 0
    # We define constraints as g(v) >= 0 in SLSQP
    constraints = []
    
    # Pair constraints: dist_ij - 2r >= 0
    for k in range(N_PAIRS):
        i, j = IU[0][k], IU[1][k]
        def c(v, i=i, j=j):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            xi = v[2*i]
            yi = v[2*i+1]
            xj = v[2*j]
            yj = v[2*j+1]
            d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            r = v[-1]
            return d - 2.0*r
        def j_c(v, i=i, j=j):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            g = np.zeros_like(v)
            xi = v[2*i]
            yi = v[2*i+1]
            xj = v[2*j]
            yj = v[2*j+1]
            d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if d < 1e-12:
                return g
            g[2*i] = (xi - xj) / d
            g[2*i+1] = (yi - yj) / d
            g[2*j] = (xj - xi) / d
            g[2*j+1] = (yj - yi) / d
            g[-1] = -2.0
            return g
        constraints.append({'type': 'ineq', 'fun': c, 'jac': j_c})
        
    # Wall constraints: d_wall - r >= 0
    for i in range(N):
        # Left: x_i + 0.5 - r >= 0
        def c_l(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return v[2*i] + HALF - v[-1]
        def j_cl(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            g = np.zeros_like(v)
            g[2*i] = 1.0
            g[-1] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_l, 'jac': j_cl})
        
        # Right: 0.5 - x_i - r >= 0
        def c_r(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return HALF - v[2*i] - v[-1]
        def j_cr(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            g = np.zeros_like(v)
            g[2*i] = -1.0
            g[-1] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_r, 'jac': j_cr})
        
        # Bottom: y_i + 0.5 - r >= 0
        def c_b(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return v[2*i+1] + HALF - v[-1]
        def j_cb(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            g = np.zeros_like(v)
            g[2*i+1] = 1.0
            g[-1] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_b, 'jac': j_cb})
        
        # Top: 0.5 - y_i - r >= 0
        def c_t(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            return HALF - v[2*i+1] - v[-1]
        def j_ct(v, i=i):
            if time.time() > deadline:
                raise RuntimeError("timeout")
            g = np.zeros_like(v)
            g[2*i+1] = -1.0
            g[-1] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': c_t, 'jac': j_ct})

    x0_flat = x0.ravel()
    r0 = 0.01 # Initial guess for r
    v0 = np.append(x0_flat, r0)
    
    bounds = [(-HALF, HALF)] * (2*N) + [(0.0, HALF)]
    
    try:
        res = minimize(obj, v0, method='SLSQP', jac=jac, bounds=bounds, 
                       constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-12})
        if res.success or res.fun < 0: # res.fun is -r, so smaller is better
            pts = res.x[:2*N].reshape(N, 2)
            r = res.x[-1]
            # Ensure points are within bounds
            pts = np.clip(pts, -HALF, HALF)
            return pts, r
    except Exception:
        pass
    return x0, _compute_r(x0)

def _hex_init(rng, jitter=0.0):
    """Hexagonal lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def construct():
    global _DEADLINE
    start = time.time()
    _DEADLINE = start + 145.0 # Safety margin
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multi-start SLSQP to find a good initial radius
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.0),
        _grid_init(rng, jitter=0.02),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > _DEADLINE - 30.0:
            break
        pts, r = _slsqp_max_radius(init, _DEADLINE - 5.0, maxiter=200)
        r_actual = _compute_r(pts)
        if r_actual > best_r:
            best_r = r_actual
            best_pts = pts.copy()
            
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
        
    # Phase 2: Binary search on radius using SLSQP feasibility
    # We check if we can place circles of radius r.
    # We use SLSQP to minimize violation for a fixed r.
    def _violation_obj(v, r_target):
        # v is 2N coords
        p = v.reshape(N, 2)
        # Wall violations
        d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                            np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
        wall_viol = np.maximum(0.0, r_target - d_wall)
        # Pair violations
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0.0, 2.0 * r_target - dist[IU])
        return np.sum(wall_viol**2) + np.sum(pair_viol**2)
        
    def _violation_grad(v, r_target):
        p = v.reshape(N, 2)
        grad = np.zeros((N, 2))
        # Wall grads
        d_left = p[:, 0] + HALF
        d_right = HALF - p[:, 0]
        d_bottom = p[:, 1] + HALF
        d_top = HALF - p[:, 1]
        walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
        min_idx = np.argmin(walls, axis=1)
        
        for i in range(N):
            viol = r_target - walls[i, min_idx[i]]
            if viol > 0:
                if min_idx[i] == 0: grad[i, 0] -= 2*viol
                elif min_idx[i] == 1: grad[i, 0] += 2*viol
                elif min_idx[i] == 2: grad[i, 1] -= 2*viol
                else: grad[i, 1] += 2*viol
                
        # Pair grads
        diff = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        for k in range(N_PAIRS):
            i, j = IU[0][k], IU[1][k]
            viol = 2.0 * r_target - dist[i, j]
            if viol > 0 and dist[i, j] > 1e-12:
                dij = p[i] - p[j]
                g = 2.0 * viol * dij / dist[i, j]
                grad[i] += g
                grad[j] -= g
        return grad.ravel()

    r_lo = best_r
    r_hi = 0.0714
    
    for _ in range(10): # Binary search iterations
        if time.time() > _DEADLINE - 10.0:
            break
        r_mid = (r_lo + r_hi) / 2.0
        
        # Try to find a configuration for r_mid
        # Start from best_pts
        x0 = best_pts.ravel()
        try:
            res = minimize(_violation_obj, x0, method='L-BFGS-B', jac=lambda v: _violation_grad(v, r_mid),
                           bounds=[(-HALF, HALF)] * (2*N),
                           options={'maxiter': 100, 'ftol': 1e-10})
            pts_test = res.x.reshape(N, 2)
            pts_test = np.clip(pts_test, -HALF, HALF)
            r_achieved = _compute_r(pts_test)
            
            if r_achieved >= r_mid - 1e-6:
                r_lo = r_mid
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_pts = pts_test.copy()
            else:
                r_hi = r_mid
        except Exception:
            r_hi = r_mid

    # Phase 3: Final Polish with Soft-Min L-BFGS
    # Use the best configuration found to polish with smooth objective
    if time.time() < _DEADLINE - 5.0:
        for beta in [50.0, 200.0, 800.0]:
            if time.time() > _DEADLINE - 2.0:
                break
            try:
                res = minimize(_soft_min_obj, best_pts.ravel(), method='L-BFGS-B',
                               jac=_soft_min_grad, args=(beta,),
                               bounds=[(-HALF, HALF)] * (2 * N),
                               options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-12})
                pts_soft = res.x.reshape(N, 2)
                pts_soft = np.clip(pts_soft, -HALF, HALF)
                r_soft = _compute_r(pts_soft)
                if r_soft > best_r:
                    best_r = r_soft
                    best_pts = pts_soft
            except Exception:
                pass

    return best_pts
