"""Circle packing in perimeter-4 rectangle: unified SLSQP (W as variable) + vectorized forces + polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 142.0

def _time_ok():
    return time.time() < DEADLINE

def _compute_distances(x, y):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.sqrt(dx * dx + dy * dy)

def _objective_fast(z):
    """Unified objective: x, y, r, w (h=2-w)."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.35 or h < 0.35:
        return 1e6

    obj = -np.sum(r)
    pw = 500.0

    # Pairwise overlap penalty (vectorized)
    d = _compute_distances(x, y)
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0, r_sum - d)
    triu_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    obj += pw * np.sum(overlap[triu_mask] ** 2)

    # Boundary penalties
    bl_x = np.maximum(0, r - x)
    br_x = np.maximum(0, x - (w - r))
    bl_y = np.maximum(0, r - y)
    br_y = np.maximum(0, y - (h - r))
    obj += pw * (np.sum(bl_x ** 2) + np.sum(br_x ** 2) +
                 np.sum(bl_y ** 2) + np.sum(br_y ** 2))

    # Aspect ratio soft penalty to discourage extreme ratios
    if w < 0.5:
        obj += pw * (0.5 - w) ** 2
    elif w > 1.5:
        obj += pw * (w - 1.5) ** 2

    return obj

def _check_validity(x, y, r, w, h, tol=1e-8):
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

def _slsqp_unified(x, y, r, w, max_iter=150):
    """SLSQP with unified variables (x, y, r, w)."""
    h = 2.0 - w
    z0 = np.zeros(3 * N + 1)
    z0[0::3][:N] = x
    z0[1::3][:N] = y
    z0[2::3][:N] = r
    z0[-1] = w

    bounds = []
    for i in range(N):
        bounds.append((0.0, 2.0))   # x
        bounds.append((0.0, 2.0))   # y
        bounds.append((1e-6, 0.5))  # r
    bounds.append((0.35, 1.65))     # w

    try:
        res = minimize(_objective_fast, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'maxls': 50})
        zo = res.x
        x_opt = zo[0::3][:N]
        y_opt = zo[1::3][:N]
        r_opt = zo[2::3][:N]
        w_opt = zo[-1]
        h_opt = 2.0 - w_opt

        # Clamp
        x_opt = np.clip(x_opt, 0.0, w_opt)
        y_opt = np.clip(y_opt, 0.0, h_opt)
        r_opt = np.maximum(r_opt, 1e-6)
        lo = np.minimum(np.minimum(x_opt, y_opt), np.minimum(w_opt - x_opt, h_opt - y_opt))
        r_opt = np.minimum(r_opt, lo)
        r_opt = np.maximum(r_opt, 1e-6)

        s = _check_validity(x_opt, y_opt, r_opt, w_opt, h_opt)
        if s is not None:
            return x_opt, y_opt, r_opt, w_opt, s
    except Exception:
        pass
    return None

def _force_relax(x, y, r, w, max_iter=200, dt=0.004):
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

        # Wall forces
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 30
            if x[i] + r[i] > w:
                fx[i] -= (x[i] + r[i] - w) * 30
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 30
            if y[i] + r[i] > h:
                fy[i] -= (y[i] + r[i] - h) * 30

        # Pairwise repulsion
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

def _polish(x, y, r, w, time_budget=10.0):
    """Iterative radius growth with position adjustments."""
    h = 2.0 - w
    x = x.copy()
    y = y.copy()
    r = r.copy()
    n = N
    start = time.time()

    for _ in range(500):
        if time.time() - start > time_budget:
            break
        if not _time_ok():
            break

        # Position adjustment
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
                    fx[i] += (dx / d[i, j]) * overlap * 10
                    fy[i] += (dy / d[i, j]) * overlap * 10
                    fx[j] -= (dx / d[i, j]) * overlap * 10
                    fy[j] -= (dy / d[i, j]) * overlap * 10

        x += 0.001 * fx
        y += 0.001 * fy
        x = np.clip(x, 0.0, w)
        y = np.clip(y, 0.0, h)

        # Try to grow radii
        d = _compute_distances(x, y)
        improved = False
        for i in range(n):
            max_growth = min(x[i] - r[i], w - x[i] - r[i],
                            y[i] - r[i], h - y[i] - r[i])
            for j in range(n):
                if i != j:
                    max_growth = min(max_growth, d[i, j] - r[i] - r[j])
            if max_growth > 1e-8:
                r[i] += min(0.001, max_growth * 0.3)
                improved = True
        if not improved:
            break

    return x, y, r

def _grid_layout(kx, ky, w, h, shrink=0.9):
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
            r_list.append(min(x, y, w - x, h - y) * 0.85)
            col += 1
        row += 1
    return np.array(cx[:N]), np.array(cy[:N]), np.array(r_list[:N])

def _random_layout(w, h):
    cx = np.random.uniform(0.1 * w, 0.9 * w, N)
    cy = np.random.uniform(0.1 * h, 0.9 * h, N)
    r = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy)) * 0.4
    r = np.maximum(r, 0.005)
    return cx, cy, r

def construct():
    np.random.seed(42)
    best_sum = 0.0
    best_result = None

    # Generate diverse initial configurations
    candidates = []
    Ws = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
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

    # Phase 1: Force relax + SLSQP on top candidates
    top_k = min(10, len(candidates))
    for idx in range(top_k):
        if not _time_ok():
            break
        s0, cx0, cy0, r0, W = candidates[idx]
        H = 2.0 - W

        cx_r, cy_r, r_r = _force_relax(cx0, cy0, r0, W, max_iter=150)
        result = _slsqp_unified(cx_r, cy_r, r_r, W, max_iter=120)
        if result is not None:
            cx_opt, cy_opt, r_opt, w_opt, s_opt = result
            if s_opt > best_sum:
                best_sum = s_opt
                best_result = (cx_opt, cy_opt, r_opt, w_opt)

    # Phase 2: More SLSQP iterations on best
    if best_result is not None and _time_ok():
        cx0, cy0, r0, w0 = best_result
        result = _slsqp_unified(cx0, cy0, r0, w0, max_iter=200)
        if result is not None:
            cx_opt, cy_opt, r_opt, w_opt, s_opt = result
            if s_opt > best_sum:
                best_sum = s_opt
                best_result = (cx_opt, cy_opt, r_opt, w_opt)

    # Phase 3: Polish
    if best_result is not None and _time_ok():
        cx0, cy0, r0, w0 = best_result
        H = 2.0 - w0
        budget = min(8.0, DEADLINE - time.time() - 2)
        if budget > 2:
            cx_p, cy_p, r_p = _polish(cx0, cy0, r0, w0, time_budget=budget)
            s_polished = _check_validity(cx_p, cy_p, r_p, w0, H)
            if s_polished is not None and s_polished > best_sum:
                best_sum = s_polished
                best_result = (cx_p, cy_p, r_p, w0)

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
