"""Circle packing in perimeter-4 rectangle: constrained SLSQP + sequential radius-growth polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 145.0

def _time_ok():
    return time.time() < DEADLINE

def _compute_distances(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _check_validity(x, y, r, w, h, tol=1e-7):
    """Return sum of radii if valid, else None."""
    if w <= 0.05 or h <= 0.05:
        return None
    for i in range(N):
        if r[i] <= 1e-6:
            return None
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return None
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return None
    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d[triu_mask] < r_sum[triu_mask] - tol):
        return None
    return float(np.sum(r))

def _build_constraints(w, h):
    """Build hard constraints for SLSQP: non-overlap + boundary."""
    cons = []
    # Non-overlap: d_ij - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def _c(x, y, r, wi, i=i, j=j):
                dx = x[wi] - x[wj]
                dy = y[wi] - y[wj]
                return np.sqrt(dx*dx + dy*dy) - r[wi] - r[wj]
            cons.append({'type': 'ineq', 'fun': _c})
    # Boundary: x_i - r_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, y, r, w, i=i: x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, y, r, w, i=i: w - x[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, y, r, w, i=i: y[i] - r[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, y, r, w, i=i: h - y[i] - r[i]})
    return cons

def _constrained_slsqp(x, y, r, w, max_iter=200):
    """SLSQP with hard constraints on positions only (radii and w fixed)."""
    h = 2.0 - w
    n = N
    x0 = np.concatenate([x, y])
    
    def obj(z):
        return -np.sum(r)  # constant, but we want to keep positions feasible
    
    # Actually, let's optimize positions to maximize minimum clearance
    def obj2(z):
        xi = z[:n]
        yi = z[n:2*n]
        # Maximize minimum clearance (sum of clearances)
        d = _compute_distances(xi, yi)
        r_sum = r[:, None] + r[None, :]
        triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        clearances = d[triu_mask] - r_sum[triu_mask]
        # Also boundary clearances
        bc = np.concatenate([xi - r, w - xi - r, yi - r, h - yi - r])
        total = np.sum(clearances) + np.sum(bc)
        return -total
    
    bounds = []
    for i in range(n):
        bounds.append((r[i], w - r[i]))
    for i in range(n):
        bounds.append((r[i], h - r[i]))
    
    # Constraints: non-overlap
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def _c(z, i=i, j=j):
                dx = z[i] - z[j]
                dy = z[n+i] - z[n+j]
                return np.sqrt(dx*dx + dy*dy) - r[i] - r[j]
            cons.append({'type': 'ineq', 'fun': _c})
    
    try:
        res = minimize(obj2, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'maxls': 100})
        zo = res.x
        x_opt = zo[:n]
        y_opt = zo[n:2*n]
        x_opt = np.clip(x_opt, r, w - r)
        y_opt = np.clip(y_opt, r, h - r)
        s = _check_validity(x_opt, y_opt, r, w, h)
        if s is not None:
            return x_opt, y_opt
    except Exception:
        pass
    return None

def _sequential_growth(x, y, r, w, time_budget=20.0):
    """Sequentially grow each circle to max feasible radius, re-optimizing positions."""
    h = 2.0 - w
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    start = time.time()
    
    for iteration in range(200):
        if time.time() - start > time_budget:
            break
        if not _time_ok():
            break
        
        improved = False
        for i in range(n):
            # Find max feasible radius for circle i
            max_r = min(x[i], w - x[i], y[i], h - y[i])
            d = _compute_distances(x, y)
            for j in range(n):
                if i != j:
                    max_r = min(max_r, d[i, j] - r[j])
            max_r = max(max_r, 1e-6)
            
            if max_r > r[i] + 1e-8:
                # Try to grow
                r[i] = max_r
                # Re-optimize positions
                result = _constrained_slsqp(x, y, r, w, max_iter=50)
                if result is not None:
                    x, y = result
                    improved = True
                else:
                    r[i] = max(r[i], 1e-6)  # revert
                    # Force relax to fix
                    x, y, r = _force_relax(x, y, r, w, max_iter=30)
        
        if not improved:
            break
    
    return x, y, r

def _force_relax(x, y, r, w, max_iter=100, dt=0.003):
    """Force-directed relaxation to resolve overlaps and boundary violations."""
    h = 2.0 - w
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
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 30
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 30
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 30
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 30

        d = _compute_distances(x, y)
        r_sum = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < r_sum[i, j] and d[i, j] > 1e-12:
                    overlap = r_sum[i, j] - d[i, j]
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    fx[i] += (dx / d[i, j]) * overlap * 15
                    fy[i] += (dy / d[i, j]) * overlap * 15
                    fx[j] -= (dx / d[i, j]) * overlap * 15
                    fy[j] -= (dy / d[i, j]) * overlap * 15

        x += dt * fx
        y += dt * fy
        x = np.clip(x, 0.0, w)
        y = np.clip(y, 0.0, h)
        lo = np.minimum(np.minimum(x, y), np.minimum(w - x, h - y))
        r = np.minimum(r, np.maximum(lo, 1e-6))

        max_f = np.max(np.abs(fx)) + np.max(np.abs(fy))
        if max_f < 1e-9:
            break

    return x, y, r

def _grid_layout(kx, ky, w, h, shrink=0.85):
    cx = np.linspace(w / (2 * kx), w * (2 * kx - 1) / (2 * kx), kx)
    cy = np.linspace(h / (2 * ky), h * (2 * ky - 1) / (2 * ky), ky)
    pts_x, pts_y, pts_r = [], [], []
    for yy in cy:
        for xx in cx:
            pts_x.append(xx)
            pts_y.append(yy)
            pts_r.append(min(xx, yy, w - xx, h - yy) * shrink)
    while len(pts_x) < N:
        pts_x.append(np.random.uniform(0.05 * w, 0.95 * w))
        pts_y.append(np.random.uniform(0.05 * h, 0.95 * h))
        pts_r.append(0.01)
    return np.array(pts_x[:N]), np.array(pts_y[:N]), np.array(pts_r[:N])

def _hex_layout(w, h):
    cx, cy, r_list = [], [], []
    base_r = min(w, h) / 10.0
    row_sp = base_r * 1.732
    col_sp = base_r * 1.5
    row = 0
    while len(cx) < N:
        y = row * row_sp
        if y > h - base_r:
            row += 1
            continue
        offset = col_sp / 2 if row % 2 else 0
        col = 0
        while len(cx) < N:
            x = col * col_sp + offset
            if x > w - base_r:
                break
            cx.append(x)
            cy.append(y)
            r_list.append(min(x, y, w - x, h - y) * 0.8)
            col += 1
        row += 1
    return np.array(cx[:N]), np.array(cy[:N]), np.array(r_list[:N])

def _random_layout(w, h):
    cx = np.random.uniform(0.15 * w, 0.85 * w, N)
    cy = np.random.uniform(0.15 * h, 0.85 * h, N)
    r = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy)) * 0.3
    r = np.maximum(r, 0.005)
    return cx, cy, r

def construct():
    np.random.seed(42)
    best_sum = 0.0
    best_result = None

    # Generate diverse initial configurations
    candidates = []
    Ws = [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
    grids = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]

    for W in Ws:
        H = 2.0 - W
        if H < 0.35 or W < 0.35:
            continue

        for kx, ky in grids:
            if kx * ky < N:
                continue
            cx, cy, r = _grid_layout(kx, ky, W, H)
            s = _check_validity(cx, cy, r, W, H)
            if s is not None:
                candidates.append((s, cx, cy, r, W))

        cx, cy, r = _hex_layout(W, H)
        s = _check_validity(cx, cy, r, W, H)
        if s is not None:
            candidates.append((s, cx, cy, r, W))

        for _ in range(2):
            cx, cy, r = _random_layout(W, H)
            s = _check_validity(cx, cy, r, W, H)
            if s is not None:
                candidates.append((s, cx, cy, r, W))

    candidates.sort(key=lambda t: -t[0])

    # Phase 1: Force relax + constrained SLSQP + sequential growth
    top_k = min(8, len(candidates))
    for idx in range(top_k):
        if not _time_ok():
            break
        s0, cx0, cy0, r0, W = candidates[idx]
        H = 2.0 - W

        cx_r, cy_r, r_r = _force_relax(cx0, cy0, r0, W, max_iter=100)
        
        # Constrained SLSQP to optimize positions
        result = _constrained_slsqp(cx_r, cy_r, r_r, W, max_iter=100)
        if result is not None:
            cx_opt, cy_opt = result
            s_opt = _check_validity(cx_opt, cy_opt, r_r, W, H)
            if s_opt is not None:
                # Sequential growth
                budget = min(15.0, DEADLINE - time.time() - 5)
                if budget > 3:
                    cx_g, cy_g, r_g = _sequential_growth(cx_opt, cy_opt, r_r, W, time_budget=budget)
                    s_g = _check_validity(cx_g, cy_g, r_g, W, H)
                    if s_g is not None and s_g > s_opt:
                        cx_opt, cy_opt, r_opt = cx_g, cy_g, r_g
                        s_opt = s_g
                else:
                    r_opt = r_r
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_result = (cx_opt, cy_opt, r_opt, W)

    # Phase 2: More sequential growth on best
    if best_result is not None and _time_ok():
        cx0, cy0, r0, w0 = best_result
        H = 2.0 - w0
        budget = min(20.0, DEADLINE - time.time() - 3)
        if budget > 3:
            cx_g, cy_g, r_g = _sequential_growth(cx0, cy0, r0, w0, time_budget=budget)
            s_g = _check_validity(cx_g, cy_g, r_g, w0, H)
            if s_g is not None and s_g > best_sum:
                best_sum = s_g
                best_result = (cx_g, cy_g, r_g, w0)

    # Fallback
    if best_result is None:
        W, H = 1.0, 1.0
        side = 5
        cx = []
        cy = []
        r_list = []
        for i in range(N):
            row, col = divmod(i, side)
            cx.append((col + 0.5) / side)
            cy.append((row + 0.5) / side)
            r_list.append(0.5 / side - 1e-4)
        best_result = (np.array(cx), np.array(cy), np.array(r_list), W)

    cx, cy, r, W = best_result
    H = 2.0 - W

    # Final clamping
    cx = np.clip(cx, 0.0, W)
    cy = np.clip(cy, 0.0, H)
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)

    out = np.column_stack([cx, cy, r])

    # Final validation
    s_final = _check_validity(out[:, 0], out[:, 1], out[:, 2], W, H)
    if s_final is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(5) for j in range(5)]
        return np.array(cells[:N])

    return out
