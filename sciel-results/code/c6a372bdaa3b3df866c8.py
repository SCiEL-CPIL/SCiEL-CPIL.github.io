import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 140.0

def _elapsed(start):
    return time.time() - start

def _sum_r(r):
    return float(np.sum(r))

def _is_valid(x, y, r, eps=1e-7):
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _make_valid(x, y, r, max_iter=500):
    """Iteratively shrink radii to satisfy all constraints."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            # Bound constraints
            r[i] = min(r[i], x[i] * 0.9999, (1.0 - x[i]) * 0.9999,
                       y[i] * 0.9999, (1.0 - y[i]) * 0.9999)
            if r[i] < 1e-8:
                r[i] = 1e-8
        # Pairwise overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-12:
                    # Shrink both proportionally
                    if min_d > 0:
                        shrink = (d / min_d) * 0.9999
                    else:
                        shrink = 0.5
                    r[i] *= shrink
                    r[j] *= shrink
                    changed = True
        if not changed:
            break
    return x, y, r

def _hex_init():
    """Hexagonal packing initialization for 32 circles."""
    rows = [6, 5, 6, 5, 6, 4]
    x = []
    y = []
    r = []
    total_rows = len(rows)
    row_h = 1.0 / total_rows
    for row_idx, count in enumerate(rows):
        ry = (row_idx + 0.5) * row_h
        offset = 0.0
        if row_idx % 2 == 1:
            offset = 0.5 / count
        for c in range(count):
            cx = (c + 0.5 + offset) / count
            cr = 0.48 * row_h
            cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, ry * 0.99, (1.0 - ry) * 0.99)
            x.append(cx)
            y.append(ry)
            r.append(cr)
    x = x[:N]
    y = y[:N]
    r = r[:N]
    while len(x) < N:
        x.append(0.5)
        y.append(0.5)
        r.append(0.02)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _rect_init():
    """Rectangular grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    x = []
    y = []
    r = []
    for i in range(side):
        for j in range(side):
            if len(x) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                cr = 0.48 / side
                cr = min(cr, cx * 0.99, (1.0 - cx) * 0.99, cy * 0.99, (1.0 - cy) * 0.99)
                x.append(cx)
                y.append(cy)
                r.append(cr)
    return np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

def _random_init(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.06)
    return x, y, r

def _phyllotaxis_init():
    """Phyllotaxis low-discrepancy initialization."""
    n = N
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)
    for i in range(n):
        r_i = np.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x[i] = 0.5 + 0.45 * r_i * np.cos(theta)
        y[i] = 0.5 + 0.45 * r_i * np.sin(theta)
        r[i] = 0.05
    r = np.minimum(r, np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)) * 0.99)
    r = np.maximum(r, 0.01)
    return x, y, r

def _objective(params, start, soft_eps=1e-4, mu=50.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start, soft_eps=1e-6, mu=200.0):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + soft_eps*soft_eps))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + soft_eps*soft_eps))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_grad(params, start):
    """Objective with analytic gradient for L-BFGS-B."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros(3*N)
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:3*N].copy()
    r_safe = np.maximum(r, 1e-10)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    dist[dist < 1e-14] = 1e-14
    min_dist = r_safe[:, None] + r_safe[None, :]
    diff = min_dist - dist
    
    eps = 1e-8
    sqrt_term = np.sqrt(diff*diff + eps)
    sqrt_term[sqrt_term < 1e-14] = 1e-14
    dsmooth = 0.5 * (1 + diff / sqrt_term)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    obj = -np.sum(r_safe)
    overlap_pen = np.sum(dsmooth[mask])
    
    dneg_x = r_safe - x
    dpos_x = x - (1.0 - r_safe)
    dneg_y = r_safe - y
    dpos_y = y - (1.0 - r_safe)
    
    sqrt_dx = np.sqrt(dneg_x*dneg_x + eps)
    sqrt_dx[sqrt_dx < 1e-14] = 1e-14
    sm_dn_x = 0.5 * (1 + dneg_x / sqrt_dx)
    
    sqrt_px = np.sqrt(dpos_x*dpos_x + eps)
    sqrt_px[sqrt_px < 1e-14] = 1e-14
    sm_dp_x = 0.5 * (1 + dpos_x / sqrt_px)
    
    sqrt_dy = np.sqrt(dneg_y*dneg_y + eps)
    sqrt_dy[sqrt_dy < 1e-14] = 1e-14
    sm_dn_y = 0.5 * (1 + dneg_y / sqrt_dy)
    
    sqrt_py = np.sqrt(dpos_y*dpos_y + eps)
    sqrt_py[sqrt_py < 1e-14] = 1e-14
    sm_dp_y = 0.5 * (1 + dpos_y / sqrt_py)
    
    bound_pen = np.sum(0.5*(dneg_x + sqrt_dx) + 0.5*(dpos_x + sqrt_px) + 
                       0.5*(dneg_y + sqrt_dy) + 0.5*(dpos_y + sqrt_py))
    
    mu = 50.0
    obj_val = obj + mu * (overlap_pen + bound_pen)
    
    grad = np.zeros(3*N)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                ds = dsmooth[i, j]
                grad[i] += -mu * ds * ux
                grad[N+i] += -mu * ds * uy
                grad[j] += mu * ds * ux
                grad[N+j] += mu * ds * uy
                grad[2*N+i] += mu * ds
                grad[2*N+j] += mu * ds
    
    grad[:N] += mu * (-sm_dn_x + sm_dp_x)
    grad[N:2*N] += mu * (-sm_dn_y + sm_dp_y)
    grad[2*N:3*N] += -1.0
    grad[2*N:3*N] += mu * (sm_dn_x + sm_dn_y)
    grad[2*N:3*N] += mu * (sm_dp_x + sm_dp_y)
    
    return obj_val, grad

def _objective_tight_grad(params, start):
    """Tight objective with analytic gradient."""
    if _elapsed(start) > TIME_LIMIT:
        return 1e10, np.zeros(3*N)
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:3*N].copy()
    r_safe = np.maximum(r, 1e-10)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    dist[dist < 1e-14] = 1e-14
    min_dist = r_safe[:, None] + r_safe[None, :]
    diff = min_dist - dist
    
    eps = 1e-10
    sqrt_term = np.sqrt(diff*diff + eps)
    sqrt_term[sqrt_term < 1e-14] = 1e-14
    dsmooth = 0.5 * (1 + diff / sqrt_term)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    obj = -np.sum(r_safe)
    overlap_pen = np.sum(dsmooth[mask])
    
    dneg_x = r_safe - x
    dpos_x = x - (1.0 - r_safe)
    dneg_y = r_safe - y
    dpos_y = y - (1.0 - r_safe)
    
    sqrt_dx = np.sqrt(dneg_x*dneg_x + eps)
    sqrt_dx[sqrt_dx < 1e-14] = 1e-14
    sm_dn_x = 0.5 * (1 + dneg_x / sqrt_dx)
    
    sqrt_px = np.sqrt(dpos_x*dpos_x + eps)
    sqrt_px[sqrt_px < 1e-14] = 1e-14
    sm_dp_x = 0.5 * (1 + dpos_x / sqrt_px)
    
    sqrt_dy = np.sqrt(dneg_y*dneg_y + eps)
    sqrt_dy[sqrt_dy < 1e-14] = 1e-14
    sm_dn_y = 0.5 * (1 + dneg_y / sqrt_dy)
    
    sqrt_py = np.sqrt(dpos_y*dpos_y + eps)
    sqrt_py[sqrt_py < 1e-14] = 1e-14
    sm_dp_y = 0.5 * (1 + dpos_y / sqrt_py)
    
    bound_pen = np.sum(0.5*(dneg_x + sqrt_dx) + 0.5*(dpos_x + sqrt_px) + 
                       0.5*(dneg_y + sqrt_dy) + 0.5*(dpos_y + sqrt_py))
    
    mu = 200.0
    obj_val = obj + mu * (overlap_pen + bound_pen)
    
    grad = np.zeros(3*N)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = dist[i, j]
            if d > 1e-14:
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                ds = dsmooth[i, j]
                grad[i] += -mu * ds * ux
                grad[N+i] += -mu * ds * uy
                grad[j] += mu * ds * ux
                grad[N+j] += mu * ds * uy
                grad[2*N+i] += mu * ds
                grad[2*N+j] += mu * ds
    
    grad[:N] += mu * (-sm_dn_x + sm_dp_x)
    grad[N:2*N] += mu * (-sm_dn_y + sm_dp_y)
    grad[2*N:3*N] += -1.0
    grad[2*N:3*N] += mu * (sm_dn_x + sm_dn_y + sm_dp_x + sm_dp_y)
    
    return obj_val, grad

def _push_apart(x, y, r, iterations=50):
    """Physics-based push apart to resolve overlaps."""
    x = x.copy()
    y = y.copy()
    for _ in range(iterations):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-10:
                    if d < 1e-10:
                        dx, dy = 1e-3, 0.0
                        d = np.hypot(dx, dy)
                    overlap = min_d - d
                    nx, ny = dx/d, dy/d
                    push = overlap * 0.5 * 0.8
                    x[i] += nx * push
                    y[i] += ny * push
                    x[j] -= nx * push
                    y[j] -= ny * push
                    moved = True
        for i in range(N):
            x[i] = np.clip(x[i], r[i]*0.99, 1.0 - r[i]*0.99)
            y[i] = np.clip(y[i], r[i]*0.99, 1.0 - r[i]*0.99)
        if not moved:
            break
    return x, y

def _coordinate_descent(params, start, time_budget=10):
    """Coordinate descent refinement on radii."""
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:3*N].copy()
    
    for _ in range(30):
        if _elapsed(start) > TIME_LIMIT - time_budget:
            break
        improved = False
        for i in range(N):
            if _elapsed(start) > TIME_LIMIT - time_budget:
                break
            old_r = r[i]
            max_r = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                max_r = min(max_r, d - r[j])
            if max_r > old_r * 1.001:
                r[i] = min(max_r * 0.9999, old_r * 1.05)
                improved = True
            elif max_r < old_r * 0.999:
                r[i] = max(max_r * 0.9999, 1e-8)
                improved = True
        if not improved:
            break
    
    return np.concatenate([x, y, r])

def _local_search(params, start_time, max_iter=300):
    """Local search with random perturbations to escape local minima."""
    best_params = params.copy()
    best_score = _sum_r(best_params[2*N:3*N])
    rng = np.random.RandomState(12345)
    
    for i in range(max_iter):
        if _elapsed(start_time) > TIME_LIMIT - 20:
            break
        
        scale = 0.02 * (1 - i / max_iter)
        perturbed = best_params + rng.normal(0, scale, size=3*N)
        
        px = perturbed[:N].copy()
        py = perturbed[N:2*N].copy()
        pr = perturbed[2*N:3*N].copy()
        px, py = _push_apart(px, py, pr, iterations=10)
        p_opt = np.concatenate([px, py, pr])
        
        try:
            res = minimize(_objective, p_opt, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxfun': 2000},
                           args=(start_time,))
            p_result = res.x
            vx = p_result[:N]
            vy = p_result[N:2*N]
            vr = p_result[2*N:3*N]
            vx, vy, vr = _make_valid(vx, vy, vr)
            score = _sum_r(vr)
            
            if score > best_score:
                best_score = score
                best_params = np.concatenate([vx, vy, vr])
        except Exception:
            continue
    
    return best_params

def _optimize(x0, start_time, time_budget=100):
    """Run L-BFGS-B optimization with gradient."""
    try:
        res = minimize(_objective_grad, x0, method='L-BFGS-B', jac=True,
                       args=(start_time,),
                       options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxfun': 50000})
        return res.x
    except Exception:
        return x0

def _tight_polish(params, start_time, time_budget=10):
    """Final high-precision polish with gradient."""
    try:
        res = minimize(_objective_tight_grad, params, method='L-BFGS-B', jac=True,
                       args=(start_time,),
                       options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14,
                                'maxfun': 100000})
        return res.x
    except Exception:
        return params

def _bisect_radius(x, y, r, tol=1e-6, max_iter=100):
    """
    Bisection on a uniform scaling factor for radii to find the maximum
    feasible uniform scale. This is a convex feasibility check.
    """
    # We want to find the largest s such that s*r_i are all feasible
    # given fixed positions x, y
    lo = 0.0
    hi = 1.0
    
    # First check if hi=1.0 is feasible
    def _check_scale(s):
        rs = s * r
        for i in range(N):
            if x[i] < rs[i] - 1e-12 or x[i] > 1.0 - rs[i] + 1e-12:
                return False
            if y[i] < rs[i] - 1e-12 or y[i] > 1.0 - rs[i] + 1e-12:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < rs[i] + rs[j] - 1e-12:
                    return False
        return True
    
    if not _check_scale(1.0):
        # Find the max feasible scale
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if _check_scale(mid):
                lo = mid
            else:
                hi = mid
            if hi - lo < tol:
                break
        s = lo
    else:
        s = 1.0
    
    return s * r

def construct():
    start_time = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal
    hx, hy, hr = _hex_init()
    inits.append(np.concatenate([hx, hy, hr]))
    
    # Rectangular
    rx, ry, rr = _rect_init()
    inits.append(np.concatenate([rx, ry, rr]))
    
    # Phyllotaxis
    px, py, pr = _phyllotaxis_init()
    inits.append(np.concatenate([px, py, pr]))
    
    # Multiple random inits
    for seed in [42, 123, 456, 789, 101, 202, 303]:
        rx2, ry2, rr2 = _random_init(seed)
        inits.append(np.concatenate([rx2, ry2, rr2]))
    
    # Perturbed hex
    rng = np.random.RandomState(7)
    for _ in range(4):
        hx2, hy2, hr2 = _hex_init()
        hx2 += rng.uniform(-0.05, 0.05, N)
        hy2 += rng.uniform(-0.05, 0.05, N)
        hx2 = np.clip(hx2, 0, 1)
        hy2 = np.clip(hy2, 0, 1)
        inits.append(np.concatenate([hx2, hy2, hr2]))
    
    # Perturbed phyllotaxis
    for _ in range(3):
        px2, py2, pr2 = _phyllotaxis_init()
        px2 += rng.uniform(-0.03, 0.03, N)
        py2 += rng.uniform(-0.03, 0.03, N)
        px2 = np.clip(px2, 0, 1)
        py2 = np.clip(py2, 0, 1)
        inits.append(np.concatenate([px2, py2, pr2]))
    
    best_params = None
    best_score = -np.inf
    
    for init_idx, x0 in enumerate(inits):
        if _elapsed(start_time) > TIME_LIMIT - 30:
            break
        
        # Push apart first
        x0a = x0[:N].copy()
        x0b = x0[N:2*N].copy()
        x0c = x0[2*N:3*N].copy()
        x0a, x0b = _push_apart(x0a, x0b, x0c, iterations=30)
        x0_opt = np.concatenate([x0a, x0b, x0c])
        
        # Optimize
        result = _optimize(x0_opt, start_time)
        
        # Make valid
        vx = result[:N]
        vy = result[N:2*N]
        vr = result[2*N:3*N]
        vx, vy, vr = _make_valid(vx, vy, vr)
        
        # Coordinate descent refinement
        params_cd = np.concatenate([vx, vy, vr])
        params_cd = _coordinate_descent(params_cd, start_time, time_budget=5)
        vx2 = params_cd[:N]
        vy2 = params_cd[N:2*N]
        vr2 = params_cd[2*N:3*N]
        vx2, vy2, vr2 = _make_valid(vx2, vy2, vr2)
        
        score = _sum_r(vr2)
        
        if score > best_score:
            best_score = score
            best_params = np.concatenate([vx2, vy2, vr2])
    
    # Local search to escape local minima
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 35:
        ls_result = _local_search(best_params, start_time, max_iter=150)
        lsx = ls_result[:N]
        lsy = ls_result[N:2*N]
        lsr = ls_result[2*N:3*N]
        lsx, lsy, lsr = _make_valid(lsx, lsy, lsr)
        ls_score = _sum_r(lsr)
        if ls_score > best_score:
            best_score = ls_score
            best_params = np.concatenate([lsx, lsy, lsr])
    
    # Final tight polish on best
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 20:
        polished = _tight_polish(best_params, start_time)
        px = polished[:N]
        py = polished[N:2*N]
        pr = polished[2*N:3*N]
        px, py, pr = _make_valid(px, py, pr)
        
        # Coordinate descent after polish
        params_cd = np.concatenate([px, py, pr])
        params_cd = _coordinate_descent(params_cd, start_time, time_budget=5)
        px2 = params_cd[:N]
        py2 = params_cd[N:2*N]
        pr2 = params_cd[2*N:3*N]
        px2, py2, pr2 = _make_valid(px2, py2, pr2)
        
        p_score = _sum_r(pr2)
        if p_score > best_score:
            best_score = p_score
            best_params = np.concatenate([px2, py2, pr2])
    
    # One more round of optimization from polished result if time permits
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 15:
        result2 = _optimize(best_params, start_time)
        vx2 = result2[:N]
        vy2 = result2[N:2*N]
        vr2 = result2[2*N:3*N]
        vx2, vy2, vr2 = _make_valid(vx2, vy2, vr2)
        s2 = _sum_r(vr2)
        if s2 > best_score:
            best_score = s2
            best_params = np.concatenate([vx2, vy2, vr2])
    
    # Final coordinate descent pass
    if best_params is not None and _elapsed(start_time) < TIME_LIMIT - 5:
        params_final = _coordinate_descent(best_params, start_time, time_budget=3)
        fx = params_final[:N]
        fy = params_final[N:2*N]
        fr = params_final[2*N:3*N]
        fx, fy, fr = _make_valid(fx, fy, fr)
        fscore = _sum_r(fr)
        if fscore > best_score:
            best_score = fscore
            best_params = np.concatenate([fx, fy, fr])
    
    if best_params is None:
        # Fallback
        x, y, r = _rect_init()
        best_params = np.concatenate([x, y, r])
    
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    result = np.column_stack((x, y, r))
    return result
