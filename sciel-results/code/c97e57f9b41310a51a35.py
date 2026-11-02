import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi
import time

N = 32
TIME_LIMIT = 140.0  # evaluator kills at 150s; keep margin

def _elapsed(start):
    return time.time() - start

def _is_valid(x, y, r, eps=1e-9):
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

def _objective(params, start, mu=50.0, smooth=1e-8):
    if _elapsed(start) > TIME_LIMIT:
        return 1e10
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    r = np.maximum(r, 0.0)
    obj = -np.sum(r)

    # Vectorized overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx*dx + dy*dy + 1e-14)
    min_dist = r[:, None] + r[None, :]
    diff = min_dist - dist
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + smooth))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    # Boundary penalties
    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + smooth))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _objective_tight(params, start, mu=200.0, smooth=1e-12):
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
    smooth_max = 0.5 * (diff + np.sqrt(diff*diff + smooth))
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlap_pen = np.sum(smooth_max[mask])

    dneg_x = r - x
    dpos_x = x - (1.0 - r)
    dneg_y = r - y
    dpos_y = y - (1.0 - r)
    def sm(v):
        return 0.5 * (v + np.sqrt(v*v + smooth))
    bound_pen = np.sum(sm(dneg_x)) + np.sum(sm(dpos_x)) + np.sum(sm(dneg_y)) + np.sum(sm(dpos_y))

    return obj + mu * (overlap_pen + bound_pen)

def _push_apart(x, y, r, iters=1000):
    """Move centers to resolve overlaps; never shrink radii. Keep in bounds."""
    x = x.copy()
    y = y.copy()
    for _ in range(iters):
        moved = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy + 1e-14)
        need = r[:, None] + r[None, :]
        overlaps = dist < need - 1e-12
        
        if not np.any(overlaps):
            break
            
        for i in range(N):
            for j in range(i + 1, N):
                if not overlaps[i, j]:
                    continue
                dxij = x[i] - x[j]
                dyij = y[i] - y[j]
                d = np.hypot(dxij, dyij)
                need_ij = r[i] + r[j]
                if d < need_ij:
                    if d < 1e-12:
                        ang = (i * 0.7 + j * 1.3) % (2*np.pi)
                        ux, uy = np.cos(ang), np.sin(ang)
                    else:
                        ux, uy = dxij / d, dyij / d
                    gap = need_ij - d
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
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(100):
        x, y = _push_apart(x, y, r, iters=500)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        if _is_valid(x, y, r):
            break
    
    if not _is_valid(x, y, r):
        for _ in range(1000):
            for i in range(N):
                mr = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
                if r[i] > mr * 0.999:
                    r[i] = max(0.0, mr * 0.998)
            
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx*dx + dy*dy + 1e-14)
            need = r[:, None] + r[None, :]
            overlaps = dist < need - 1e-10
            
            if not np.any(overlaps):
                break
            
            idx = np.where(overlaps)
            i, j = idx[0][0], idx[1][0]
            if j < i:
                i, j = j, i
            r[i] *= 0.99
            r[j] *= 0.99
    
    return x, y, r

def _hex_init():
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

def _voronoi_init():
    rng = np.random.RandomState(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.zeros(N)
    for i in range(N):
        dists = np.sqrt((x[i]-x)**2 + (y[i]-y)**2)
        dists[i] = np.inf
        nearest = np.argsort(dists)[:3]
        min_dist = np.min(dists[nearest])
        r[i] = 0.45 * min_dist
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    return x, y, r

def _grid_init(rows, cols):
    """Create a grid with exactly N circles, distributing extras."""
    cells = []
    for j in range(rows):
        for i in range(cols):
            if len(cells) < N:
                x = (i + 0.5) / cols
                y = (j + 0.5) / rows
                cells.append([x, y])
    x = np.array([c[0] for c in cells])
    y = np.array([c[1] for c in cells])
    # Compute radius based on local spacing
    r = np.zeros(N)
    for i in range(N):
        mn = 1e9
        for j in range(N):
            if i != j:
                mn = min(mn, np.hypot(x[i]-x[j], y[i]-y[j]))
        r[i] = 0.49 * mn
        r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
    return x, y, r

def _optimize(x, y, r, start):
    params = np.concatenate([x, y, r])
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    # L-BFGS-B with penalty
    if _elapsed(start) < TIME_LIMIT - 20:
        try:
            bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.45)] * N
            res = minimize(
                _objective, params, args=(start, 50.0, 1e-8),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
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
    if _elapsed(start) < TIME_LIMIT - 10:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start, 100.0, 1e-10),
                method='Powell',
                options={'maxiter': 3000, 'ftol': 1e-12}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Nelder-Mead
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start, 50.0, 1e-8),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
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

def _tight_polish(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    # Powell with tight objective
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective_tight, p0, args=(start, 200.0, 1e-12),
                method='Powell',
                options={'maxiter': 5000, 'ftol': 1e-14}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # L-BFGS-B with tight objective
    if _elapsed(start) < TIME_LIMIT - 3:
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.45)] * N
            res = minimize(
                _objective_tight, p0, args=(start, 200.0, 1e-12),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Fine-grained local perturbation
    if _elapsed(start) < TIME_LIMIT - 1:
        rng = np.random.RandomState(42)
        for trial in range(200):
            if _elapsed(start) > TIME_LIMIT - 1:
                break
            scale = 0.003 * (1.0 - trial / 200.0)
            px = best[0] + rng.normal(0, scale, N)
            py = best[1] + rng.normal(0, scale, N)
            pr = best[2] + rng.normal(0, scale * 0.2, N)
            pr = np.maximum(pr, 1e-4)
            
            vx, vy, vr = _make_valid(px, py, pr)
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
    
    return best

def _final_polish(x, y, r, start):
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    # Multiple Powell passes with decreasing tolerance
    for ftol in [1e-10, 1e-12, 1e-14]:
        if _elapsed(start) > TIME_LIMIT - 8:
            break
        try:
            p0 = np.concatenate([best[0], best[1], best[2]])
            res = minimize(
                _objective, p0, args=(start, 100.0, 1e-10),
                method='Powell',
                options={'maxiter': 3000, 'ftol': ftol}
            )
            vx, vy, vr = _make_valid(res.x[:N], res.x[N:2*N], res.x[2*N:3*N])
            if _is_valid(vx, vy, vr):
                s = _sum_r(vr)
                if s > best_score:
                    best_score = s
                    best = (vx, vy, vr)
        except Exception:
            pass
    
    # Tight polish
    if _elapsed(start) < TIME_LIMIT - 5:
        try:
            res = _tight_polish(*best, start)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass
    
    return best

def _physics_optimize(x, y, r, start, max_iter=3000):
    """
    Physics-based optimization using repulsive particle model.
    Optimizes positions and radii simultaneously with gradient descent.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    soft = 1e-6
    
    def compute_forces(x, y, r):
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy) + soft
                ux = dx / d
                uy = dy / d
                
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    F = 50.0 * overlap / d
                else:
                    F = -0.3 * (1.0 / (d * d))
                
                fx[i] += F * ux
                fy[i] += F * uy
                fx[j] -= F * ux
                fy[j] -= F * uy
                
                if overlap > 0:
                    fr[i] -= 10.0
                    fr[j] -= 10.0
                else:
                    fr[i] += 0.05 / (d + 1.0)
                    fr[j] += 0.05 / (d + 1.0)
        
        for i in range(N):
            dist_left = x[i] - r[i]
            if dist_left < 0.05:
                fx[i] += 20.0 * (0.05 - dist_left)
                fr[i] -= 5.0 * (0.05 - dist_left)
            dist_right = (1.0 - x[i]) - r[i]
            if dist_right < 0.05:
                fx[i] -= 20.0 * (0.05 - dist_right)
                fr[i] -= 5.0 * (0.05 - dist_right)
            dist_bottom = y[i] - r[i]
            if dist_bottom < 0.05:
                fy[i] += 20.0 * (0.05 - dist_bottom)
                fr[i] -= 5.0 * (0.05 - dist_bottom)
            dist_top = (1.0 - y[i]) - r[i]
            if dist_top < 0.05:
                fy[i] -= 20.0 * (0.05 - dist_top)
                fr[i] -= 5.0 * (0.05 - dist_top)
        
        return fx, fy, fr
    
    step_size = 0.005
    best_score = _sum_r(r)
    best_state = (x.copy(), y.copy(), r.copy())
    patience = 0
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 10:
            break
        
        fx, fy, fr = compute_forces(x, y, r)
        
        x += step_size * fx
        y += step_size * fy
        r += step_size * fr
        
        r = np.maximum(r, 1e-4)
        r = np.minimum(r, 0.45)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        score = _sum_r(r)
        if score > best_score + 1e-8:
            best_score = score
            best_state = (x.copy(), y.copy(), r.copy())
            patience = 0
        else:
            patience += 1
        
        if patience > 100:
            step_size *= 0.95
            patience = 0
            if step_size < 1e-7:
                break
        elif patience > 30:
            step_size *= 0.98
    
    bx, by, br = best_state
    bx, by, br = _make_valid(bx, by, br)
    return bx, by, br

def _anneal_optimize(x, y, r, start, max_iter=5000):
    """Simulated annealing style optimization for escaping local minima."""
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    best = (x.copy(), y.copy(), r.copy())
    best_score = _sum_r(r)
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if _elapsed(start) > TIME_LIMIT - 15:
            break
        
        # Random perturbation
        scale = T * 10
        px = x + rng.normal(0, scale, N)
        py = y + rng.normal(0, scale, N)
        pr = r + rng.normal(0, scale * 0.3, N)
        pr = np.maximum(pr, 1e-4)
        pr = np.minimum(pr, 0.45)
        
        # Quick validity check
        if not _is_valid(px, py, pr):
            # Try to fix
            px, py, pr = _make_valid(px, py, pr)
        
        if _is_valid(px, py, pr):
            s_new = _sum_r(pr)
            s_curr = _sum_r(r)
            
            if s_new > s_curr or rng.random() < np.exp(-(s_curr - s_new) / T):
                x, y, r = px, py, pr
                if s_new > best_score:
                    best_score = s_new
                    best = (x.copy(), y.copy(), r.copy())
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best

def construct():
    start = time.time()
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_init())
    inits.append(_rect_init(6, 6))
    inits.append(_rect_init(7, 5))
    inits.append(_rect_init(5, 7))
    inits.append(_rect_init(8, 4))
    inits.append(_rect_init(4, 8))
    inits.append(_rect_init(9, 4))
    inits.append(_rect_init(4, 9))
    inits.append(_rect_init(10, 4))
    inits.append(_rect_init(4, 10))
    inits.append(_grid_init(4, 8))
    inits.append(_grid_init(8, 4))
    inits.append(_random_init(1))
    inits.append(_random_init(7))
    inits.append(_random_init(13))
    inits.append(_random_init(23))
    inits.append(_voronoi_init())
    
    # Optimize each initialization
    best = inits[0]
    best_score = _sum_r(best[2])
    if not _is_valid(*best):
        best = _make_valid(*best)
        best_score = _sum_r(best[2])

    for k, (x, y, r) in enumerate(inits):
        if _elapsed(start) > TIME_LIMIT - 30:
            break
        
        # Quick validity check and fix
        if not _is_valid(x, y, r):
            x, y, r = _make_valid(x, y, r)
        
        # Physics-based optimization for promising starts
        if _elapsed(start) < TIME_LIMIT - 25:
            try:
                vx, vy, vr = _physics_optimize(x, y, r, start, max_iter=500)
                if _is_valid(vx, vy, vr):
                    s = _sum_r(vr)
                    if s > best_score:
                        best_score = s
                        best = (vx, vy, vr)
            except Exception:
                pass
        
        # Standard optimization
        res = _optimize(x, y, r, start)
        if _is_valid(*res):
            s = _sum_r(res[2])
            if s > best_score:
                best_score = s
                best = res

    # Simulated annealing from best
    if _elapsed(start) < TIME_LIMIT - 20:
        try:
            res = _anneal_optimize(*best, start, max_iter=2000)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass

    # Final high-precision polish
    if _elapsed(start) < TIME_LIMIT - 8:
        try:
            res = _final_polish(*best, start)
            if _is_valid(*res):
                s = _sum_r(res[2])
                if s > best_score:
                    best_score = s
                    best = res
        except Exception:
            pass

    x, y, r = best
    x, y, r = _make_valid(x, y, r)
    result = np.column_stack((x, y, r))
    return result
