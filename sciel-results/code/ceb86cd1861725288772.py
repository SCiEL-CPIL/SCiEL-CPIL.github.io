import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 140.0  # evaluator kills at 150s; keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-7):
    # bounds
    if np.any(x < -eps) or np.any(x > 1.0 + eps):
        return False
    if np.any(y < -eps) or np.any(y > 1.0 + eps):
        return False
    if np.any(r < -eps):
        return False
    # circle in square
    if np.any(x < r - eps) or np.any(x > 1.0 - r + eps):
        return False
    if np.any(y < r - eps) or np.any(y > 1.0 - r + eps):
        return False
    # pairwise disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            if d < (r[i] + r[j]) - eps:
                return False
    return True

def _sum_r(r):
    return float(np.sum(r))

def _objective(params, start, mu=100.0):
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
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-8))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-8))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start, mu=500.0):
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
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-10))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + 1e-10))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iters=400):
    """Move centers to resolve overlaps; never shrink radii. Keep in bounds."""
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                need = r[i] + r[j]
                if d < need:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dx / d, dy / d
                    gap = need - d
                    step = 0.5 * gap
                    x[i] += step * ux
                    y[i] += step * uy
                    x[j] -= step * ux
                    y[j] -= step * uy
                    moved = True
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if not moved:
            break
    return x, y

def _make_valid(x, y, r):
    """Guarantee validity: push apart, then shrink any residual overlap/bounds."""
    x, y = _push_apart(x, y, r)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    for _ in range(200):
        ok = True
        for i in range(N):
            mx = min(x[i], 1.0 - x[i])
            my = min(y[i], 1.0 - y[i])
            mr = min(mx, my)
            if r[i] > mr:
                r[i] = max(0.0, mr * 0.999)
                ok = False
        if not ok:
            continue
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                need = r[i] + r[j]
                if d < need:
                    f = 0.995
                    r[i] *= f
                    r[j] *= f
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
    return x, y, r

def _hex_init():
    # Hexagonal packing for 32 circles
    pts = []
    rowh = np.sqrt(3.0)/2.0
    r = 0
    while len(pts) < N:
        for c in range(0, 12):
            x = c + (0.5 if (r % 2) else 0.0)
            y = r * rowh
            pts.append([x, y])
            if len(pts) >= N:
                break
        r += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin)
    if span < 1e-9:
        span = 1.0
    scale = 0.85 / span
    x = (pts[:,0] - xmin) * scale + 0.075
    y = (pts[:,1] - ymin) * scale + 0.075
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _rect_init(cols, rows):
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    r = np.full(N, 0.5 / max(cols, rows) * 0.98)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _random_init(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.15, 0.85, N)
    y = rng.uniform(0.15, 0.85, N)
    r = np.full(N, 0.07)
    r = np.minimum(r, np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y)))
    return x, y, r

def _tight_polish(x, y, r, start):
    """High-precision polish using tight objective and SLSQP with explicit constraints."""
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # First, try Powell with tight objective
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            res = minimize(
                _objective_tight, params, args=(start,),
                method='Powell',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Then, try SLSQP with explicit constraints for more precise local optimization
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            
            def obj_slsqp(p):
                return _objective_tight(p, start, mu=1000.0)
            
            # Constraints: each circle must be inside the square
            cons = []
            for i in range(N):
                # x >= r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
                # x <= 1-r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[2*N+i]) - p[i]})
                # y >= r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
                # y <= 1-r
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: (1.0 - p[2*N+i]) - p[N+i]})
                # r >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i]})
            
            # Pairwise constraints: distance >= r_i + r_j
            for i in range(N):
                for j in range(i+1, N):
                    def pair_con(p, i=i, j=j):
                        dx = p[i] - p[j]
                        dy = p[N+i] - p[N+j]
                        d = np.sqrt(dx*dx + dy*dy + 1e-14)
                        return d - (p[2*N+i] + p[2*N+j])
                    cons.append({'type': 'ineq', 'fun': pair_con})
            
            bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
            
            res = minimize(
                obj_slsqp, p0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    return best

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x, y, r)
    best_score = _sum_r(r)
    
    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            res = minimize(
                _objective, params, args=(start,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Powell polish
    if _elapsed(start) < TIME_LIMIT - 2:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start,),
                method='Powell',
                options={'maxiter': 800, 'ftol': 1e-10}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    return best

def construct():
    start = time.time()
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_rect_init(9, 4))
    inits.append(_rect_init(4, 9))
    inits.append(_random_init(1))
    inits.append(_random_init(7))

    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 8:
            break
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Final tight polish on best
    if _elapsed(start) < TIME_LIMIT - 4:
        res = _tight_polish(*best, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best = res

    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
