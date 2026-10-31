"""Circle packing in perimeter-4 rectangle.

Combines the topological contact-graph model (current program) with the
robust SLSQP + force-directed relaxation machinery (inspiration 1), plus a
hexagonal initial layout that matches the natural dense packing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _valid(cx, cy, r, W, H):
    """Check circles fit in [0,W]x[0,H] with disjoint interiors."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx - r < -1e-9) or np.any(cy - r < -1e-9):
        return False
    if np.any(cx + r > W + 1e-9) or np.any(cy + r > H + 1e-9):
        return False
    n = len(cx)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                return False
    return True


def _force_relax(cx, cy, r, W, H, max_iter=1500, dt=0.0008):
    """Force-directed relaxation to remove overlaps and wall violations."""
    n = len(cx)
    cx = cx.copy()
    cy = cy.copy()
    for _ in range(max_iter):
        fx = np.zeros(n)
        fy = np.zeros(n)
        for i in range(n):
            if cx[i] - r[i] < 0:
                fx[i] += (cx[i] - r[i]) * 10.0
            if cy[i] - r[i] < 0:
                fy[i] += (cy[i] - r[i]) * 10.0
            if cx[i] + r[i] > W:
                fx[i] -= (cx[i] + r[i] - W) * 10.0
            if cy[i] + r[i] > H:
                fy[i] -= (cy[i] + r[i] - H) * 10.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d2 = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d2 < min_d * min_d and d2 > 1e-14:
                    d = np.sqrt(d2)
                    overlap = min_d - d
                    ux = dx / d
                    uy = dy / d
                    fx[i] += ux * overlap * 5.0
                    fy[i] += uy * overlap * 5.0
                    fx[j] -= ux * overlap * 5.0
                    fy[j] -= uy * overlap * 5.0
        cx += dt * fx
        cy += dt * fy
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    return cx, cy


def _hex_layout(n, W, H, seed=0):
    """Build a hexagonal (honeycomb) initial layout scaled into the box."""
    rng = np.random.default_rng(seed)
    # hexagonal lattice: rows offset, vertical spacing = sqrt(3)/2 * spacing
    cols = int(np.ceil(np.sqrt(n)))
    spacing = 1.0
    verts = spacing * np.sqrt(3.0) / 2.0
    pts = []
    row = 0
    while len(pts) < n:
        xoff = (0.5 * spacing if row % 2 else 0.0)
        for col in range(cols):
            if len(pts) < n:
                pts.append((col * spacing + xoff, row * verts))
        row += 1
    pts = np.array(pts[:n], dtype=float)
    # normalize to unit square
    xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
    xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
    sx = (xmax - xmin) if xmax > xmin else 1.0
    sy = (ymax - ymin) if ymax > ymin else 1.0
    pts[:, 0] = (pts[:, 0] - xmin) / sx
    pts[:, 1] = (pts[:, 1] - ymin) / sy
    # scale into box with small margin
    m = 0.9
    cx = (pts[:, 0] * (W * m)) + W * (1 - m) / 2
    cy = (pts[:, 1] * (H * m)) + H * (1 - m) / 2
    base_r = 0.5 * spacing * min(W / sx, H / sy) * 0.42
    r = np.full(n, base_r)
    return cx, cy, r


def _grid_layout(n, W, H, seed=0):
    """Build a grid initial layout scaled into the box."""
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cx = np.array([(i + 0.5) / cols for i in range(cols) for _ in range(rows)][:n]) * W
    cy = np.array([(j + 0.5) / rows for _ in range(cols) for j in range(rows)][:n]) * H
    r = np.full(n, 0.5 * min(W / cols, H / rows) * 0.85)
    return cx, cy, r


def _optimize(W, H, layout, max_iter=600, time_budget=20.0):
    """Optimize positions and radii for a fixed rectangle W x H."""
    if layout == 'hex':
        cx, cy, r = _hex_layout(N, W, H, seed=int(W * 1000) + int(H * 1000))
    else:
        cx, cy, r = _grid_layout(N, W, H, seed=int(W * 1000) + int(H * 1000))

    # relax to a valid starting point
    cx, cy = _force_relax(cx, cy, r, W, H, max_iter=800)
    # shrink radii slightly to guarantee validity
    r = r * 0.98
    cx, cy = _force_relax(cx, cy, r, W, H, max_iter=300)

    n = N
    idx = np.arange(n)

    def objective(x):
        cxv = x[:n]
        cyv = x[n:2 * n]
        rv = x[2 * n:3 * n]
        pen = 0.0
        # wall penalties
        pen += np.sum(np.minimum(0.0, cxv - rv) ** 2)
        pen += np.sum(np.minimum(0.0, cyv - rv) ** 2)
        pen += np.sum(np.maximum(0.0, cxv + rv - W) ** 2)
        pen += np.sum(np.maximum(0.0, cyv + rv - H) ** 2)
        # pairwise overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dx = cxv[i] - cxv[j]
                dy = cyv[i] - cyv[j]
                d2 = dx * dx + dy * dy
                s = rv[i] + rv[j]
                if d2 < s * s:
                    d = np.sqrt(d2) if d2 > 1e-14 else 0.0
                    pen += (s - d) ** 2
        return -np.sum(rv) + 20.0 * pen

    x0 = np.concatenate([cx, cy, r])
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))
    for i in range(n):
        bounds.append((0.0, H))
    for i in range(n):
        bounds.append((1e-4, 0.5 * min(W, H)))

    t0 = time.time()
    best_x = x0.copy()
    best_f = objective(x0)
    for attempt in range(3):
        if time.time() - t0 > time_budget:
            break
        x_cur = x0.copy()
        if attempt > 0:
            rng = np.random.default_rng(attempt * 77 + 3)
            x_cur = x_cur + rng.uniform(-0.005, 0.005, size=x0.shape)
        try:
            res = minimize(objective, x_cur, method='SLSQP', bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-9})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    cxv = best_x[:n]
    cyv = best_x[n:2 * n]
    rv = best_x[2 * n:3 * n]
    if _valid(cxv, cyv, rv, W, H):
        return cxv, cyv, rv, float(np.sum(rv))
    # try to repair
    cxv, cyv = _force_relax(cxv, cyv, rv, W, H, max_iter=400)
    if _valid(cxv, cyv, rv, W, H):
        return cxv, cyv, rv, float(np.sum(rv))
    return None, None, None, -float('inf')


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_val = -float('inf')
    best_result = None

    # W + H = 2 (perimeter 4). Explore a range of aspect ratios.
    W_list = [1.0, 1.2, 0.85, 1.35, 0.75, 1.5, 0.65, 1.6, 0.6, 1.75, 0.5, 1.9, 0.4]

    for W in W_list:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        # per-aspect budget, capped
        budget = min(18.0, remaining * 0.9)
        for layout in ['hex', 'grid']:
            if time.time() > deadline:
                break
            cx, cy, r, val = _optimize(W, H, layout, max_iter=500,
                                       time_budget=budget)
            if val > best_val:
                best_val = val
                best_result = (cx, cy, r)

    # fallback: simple square grid
    if best_result is None or best_val <= 0:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    cx, cy, r = best_result
    return np.column_stack([cx, cy, r])
