"""Circle packing in perimeter-4 rectangle: unified SLSQP with w,h,r as decision vars + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0

def _time_ok():
    return time.time() < DEADLINE

def _dist_matrix(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _is_valid(x, y, r, w, h, tol=1e-6):
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(N):
        if r[i] <= 1e-6:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    d = _dist_matrix(x, y)
    rs = r[:, None] + r[None, :]
    triu = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[triu] < rs[triu] - tol):
        return False
    return True

def _sum_r(x, y, r, w, h):
    if not _is_valid(x, y, r, w, h):
        return 0.0
    return float(np.sum(r))

def _build_cons(w, h):
    """Inequality constraints: non-overlap + boundary. Variables: [x,y,r,w,h] (2N+2)."""
    cons = []
    n = N
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            def _c(z, i=i, j=j):
                xi, xj = z[i], z[j]
                yi, yj = z[n + i], z[n + j]
                ri, rj = z[2 * n + i], z[2 * n + j]
                dx = xi - xj
                dy = yi - yj
                return np.sqrt(dx * dx + dy * dy) - ri - rj
            cons.append({'type': 'ineq', 'fun': _c})
    # Boundary
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[4 * n + 1] - z[i] - z[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[n + i] - z[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[4 * n + 2] - z[n + i] - z[2 * n + i]})
    return cons

def _slsqp_unified(x0, y0, r0, w0, h0, max_iter=300):
    """Unified SLSQP: optimize x,y,r,w,h jointly."""
    n = N
    z0 = np.concatenate([x0, y0, r0, [w0, h0]])
    
    def obj(z):
        xi = z[:n]
        yi = z[n:2*n]
        ri = z[2*n:3*n]
        wi = z[3*n]
        hi = z[3*n+1]
        # Perimeter constraint handled by bounds
        # Maximize sum of radii
        return -np.sum(ri)
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1.5))  # x
    for i in range(n):
        bounds.append((0, 1.5))  # y
    for i in range(n):
        bounds.append((1e-4, 0.5))  # r
    bounds.append((0.3, 1.7))  # w
    bounds.append((0.3, 1.7))  # h
    
    cons = _build_cons(w0, h0)
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50})
        zo = res.x
        x_opt = zo[:n]
        y_opt = zo[n:2*n]
        r_opt = zo[2*n:3*n]
        w_opt = zo[3*n]
        h_opt = zo[3*n+1]
        # Ensure perimeter
        per = 2 * (w_opt + h_opt)
        if per > 4.0:
            scale = 4.0 / per
            x_opt *= scale
            y_opt *= scale
            r_opt *= scale
            w_opt *= scale
            h_opt *= scale
        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return x_opt, y_opt, r_opt, w_opt, h_opt
    except Exception:
        pass
    return None

def _force_relax(x, y, r, w, h, max_iter=200, dt=0.002):
    """Force-directed relaxation."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    for _ in range(max_iter):
        if not _time_ok():
            break
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 50
            if x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 50
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 50
            if y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 50
        d = _dist_matrix(x, y)
        rs = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < rs[i, j] and d[i, j] > 1e-12:
                    ov = rs[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    fx[i] += (dx / d[i, j]) * ov * 20
                    fy[i] += (dy / d[i, j]) * ov * 20
                    fx[j] -= (dx / d[i, j]) * ov * 20
                    fy[j] -= (dy / d[i, j]) * ov * 20
        x += dt * fx
        y += dt * fy
        x = np.clip(x, 0, w)
        y = np.clip(y, 0, h)
        lo = np.minimum(np.minimum(x, y), np.minimum(w - x, h - y))
        r = np.minimum(r, np.maximum(lo, 1e-4))
        if np.max(np.abs(fx)) + np.max(np.abs(fy)) < 1e-8:
            break
    return x, y, r

def _polish(x, y, r, w, h, time_budget=10.0):
    """Sequential radius growth polish."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    start = time.time()
    for it in range(500):
        if time.time() - start > time_budget or not _time_ok():
            break
        improved = False
        for i in range(n):
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            d = _dist_matrix(x, y)
            for j in range(n):
                if i != j:
                    max_r = min(max_r, d[i, j] - r[j])
            max_r = max(max_r, 1e-4)
            if max_r > r[i] + 1e-7:
                r[i] = max_r
                # Quick relax
                x2, y2, r2 = _force_relax(x, y, r, w, h, max_iter=30, dt=0.001)
                if _is_valid(x2, y2, r2, w, h):
                    x, y, r = x2, y2, r2
                    improved = True
                else:
                    r[i] = max(r[i], 1e-4)
                    x, y, r = _force_relax(x, y, r, w, h, max_iter=20, dt=0.001)
        if not improved:
            break
    return x, y, r

def _seed_grid(kx, ky, w, h):
    """Generate grid layout."""
    cx = np.linspace(w / (2 * kx), w * (2 * kx - 1) / (2 * kx), kx)
    cy = np.linspace(h / (2 * ky), h * (2 * ky - 1) / (2 * ky), ky)
    pts_x, pts_y, pts_r = [], [], []
    for yy in cy:
        for xx in cx:
            pts_x.append(xx)
            pts_y.append(yy)
            pts_r.append(min(xx, yy, w - xx, h - yy) * 0.9)
    while len(pts_x) < N:
        pts_x.append(np.random.uniform(0.1 * w, 0.9 * w))
        pts_y.append(np.random.uniform(0.1 * h, 0.9 * h))
        pts_r.append(0.01)
    return np.array(pts_x[:N]), np.array(pts_y[:N]), np.array(pts_r[:N])

def _seed_random(w, h):
    cx = np.random.uniform(0.2 * w, 0.8 * w, N)
    cy = np.random.uniform(0.2 * h, 0.8 * h, N)
    r = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy)) * 0.4
    r = np.maximum(r, 0.01)
    return cx, cy, r

def construct():
    np.random.seed(123)
    best_sum = 0.0
    best = None
    
    # Try multiple aspect ratios
    aspects = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4]
    candidates = []
    
    for ar in aspects:
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        if w < 0.4 or h < 0.4:
            continue
        # Grid seeds
        for kx, ky in [(3, 3), (4, 3), (3, 4), (5, 3), (3, 5), (4, 4)]:
            if kx * ky < N:
                continue
            cx, cy, r = _seed_grid(kx, ky, w, h)
            s = _sum_r(cx, cy, r, w, h)
            if s > 0:
                candidates.append((s, cx, cy, r, w, h))
        # Random seeds
        for _ in range(3):
            cx, cy, r = _seed_random(w, h)
            s = _sum_r(cx, cy, r, w, h)
            if s > 0:
                candidates.append((s, cx, cy, r, w, h))
    
    candidates.sort(key=lambda t: -t[0])
    
    # Process top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not _time_ok():
            break
        s0, cx0, cy0, r0, w0, h0 = candidates[idx]
        
        # Force relax first
        cx_r, cy_r, r_r = _force_relax(cx0, cy0, r0, w0, h0, max_iter=150)
        
        # Unified SLSQP
        result = _slsqp_unified(cx_r, cy_r, r_r, w0, h0, max_iter=200)
        if result is not None:
            cx_s, cy_s, r_s, w_s, h_s = result
            s_s = _sum_r(cx_s, cy_s, r_s, w_s, h_s)
            if s_s > 0:
                # Polish
                budget = min(8.0, DEADLINE - time.time() - 5)
                if budget > 2:
                    cx_p, cy_p, r_p = _polish(cx_s, cy_s, r_s, w_s, h_s, time_budget=budget)
                    s_p = _sum_r(cx_p, cy_p, r_p, w_s, h_s)
                    if s_p > s_s:
                        cx_s, cy_s, r_s = cx_p, cy_p, r_p
                        s_s = s_p
                if s_s > best_sum:
                    best_sum = s_s
                    best = (cx_s, cy_s, r_s, w_s, h_s)
    
    # If no good result, fallback
    if best is None:
        w, h = 1.0, 1.0
        side = 5
        cx = []
        cy = []
        r_list = []
        for i in range(N):
            row, col = divmod(i, side)
            cx.append((col + 0.5) / side)
            cy.append((row + 0.5) / side)
            r_list.append(0.5 / side - 1e-3)
        best = (np.array(cx), np.array(cy), np.array(r_list), w, h)
        best_sum = _sum_r(*best)
    
    cx, cy, r, w, h = best
    
    # Final clamping
    cx = np.clip(cx, 0.0, w)
    cy = np.clip(cy, 0.0, h)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    if not _is_valid(out[:, 0], out[:, 1], out[:, 2], w, h):
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(5) for j in range(5)]
        return np.array(cells[:N])
    
    return out
