"""Circle packing in unit square: maximize sum of radii of 26 disjoint circles in [0,1]^2."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_START = time.time()
_EPS = 1e-9


def _remaining():
    return _DEADLINE - (time.time() - _START) - 5.0


def _feasible_score(pos, rad):
    """Return (value, feasible)."""
    n = len(rad)
    if np.any(rad < -_EPS) or np.any(pos[:, 0] - rad < -_EPS) or np.any(pos[:, 0] + rad > 1 + _EPS) \
       or np.any(pos[:, 1] - rad < -_EPS) or np.any(pos[:, 1] + rad > 1 + _EPS):
        return -np.inf, False
    dx = pos[:, 0][:, None] - pos[:, 0][None, :]
    dy = pos[:, 1][:, None] - pos[:, 1][None, :]
    d2 = dx * dx + dy * dy
    need = rad[:, None] + rad[None, :]
    iu = np.triu_indices(n, 1)
    if np.any(d2[iu] < need[iu] ** 2 - _EPS):
        return -np.inf, False
    return float(np.sum(rad)), True


def _project_feasible(pos, rad, iters=50):
    """Heuristic repair: shrink radii / nudge to satisfy constraints."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    for _ in range(iters):
        wall_changed = False
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            r = min(r, x, 1.0 - x, y, 1.0 - y)
            r = max(r, 0.0)
            x = min(max(x, r), 1.0 - r)
            y = min(max(y, r), 1.0 - r)
            if pos[i, 0] != x or pos[i, 1] != y or rad[i] != r:
                wall_changed = True
            pos[i, 0], pos[i, 1], rad[i] = x, y, r
        overlap_changed = False
        for _ in range(5):
            inner_changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    need = rad[i] + rad[j]
                    if d < need - _EPS:
                        s = (d - _EPS) / (need + _EPS) if need > 0 else 1.0
                        rad[i] = max(rad[i] * s, 0.0)
                        rad[j] = max(rad[j] * s, 0.0)
                        inner_changed = True
            if not inner_changed:
                break
            overlap_changed = True
        if not wall_changed and not overlap_changed:
            break
    val, feas = _feasible_score(pos, rad)
    return pos, rad, feas


def _grow_radii(pos, rad, max_iters=80):
    """Grow radii per-circle: for each circle, find max radius given fixed positions and other radii."""
    pos = pos.copy()
    rad = rad.copy()
    n = len(rad)
    best_rad = rad.copy()
    best_val = float(np.sum(rad))
    for _ in range(max_iters):
        improved = False
        for i in range(n):
            r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                r_from_j = d - rad[j]
                if r_from_j < r_max:
                    r_max = r_from_j
            r_max = max(r_max, 0.0)
            if r_max > rad[i] + 1e-9:
                rad[i] = r_max
                improved = True
        if not improved:
            break
        val = float(np.sum(rad))
        if val > best_val:
            best_val = val
            best_rad = rad.copy()
    return pos, best_rad


def _soft_objective(z, lam):
    """z = [x0,y0,r0, x1,y1,r1, ...]. Minimize -sum(r) + lam*(overlap + wall penalties)."""
    n = N
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    base = -np.sum(r)
    pen = 0.0
    pen += np.sum(np.maximum(r - x, 0.0) ** 2)
    pen += np.sum(np.maximum(x + r - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(r - y, 0.0) ** 2)
    pen += np.sum(np.maximum(y + r - 1.0, 0.0) ** 2)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        pen += np.sum((needu[mask] - du) ** 2)
    return base + lam * pen


def _soft_grad(z, lam):
    n = N
    x = z[0::3].copy()
    y = z[1::3].copy()
    r = z[2::3].copy()
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    gr -= 1.0
    m = r - x
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = x + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gx[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    m = r - y
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos] * (-1.0)
        gr[mpos] += lam * 2.0 * m[mpos]
    m = y + r - 1.0
    mpos = m > 0
    if np.any(mpos):
        gy[mpos] += lam * 2.0 * m[mpos]
        gr[mpos] += lam * 2.0 * m[mpos]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    need = r[:, None] + r[None, :]
    iu = np.triu_indices(n, 1)
    d2u = d2[iu]
    needu = need[iu]
    mask = d2u < needu * needu
    if np.any(mask):
        idx_i = iu[0][mask]
        idx_j = iu[1][mask]
        dxu = dx[iu][mask]
        dyu = dy[iu][mask]
        du = np.sqrt(np.maximum(d2u[mask], 1e-12))
        fu = needu[mask] - du
        inv_u = 1.0 / du
        np.add.at(gx, idx_i, lam * 2.0 * fu * (-dxu * inv_u))
        np.add.at(gx, idx_j, lam * 2.0 * fu * (dxu * inv_u))
        np.add.at(gy, idx_i, lam * 2.0 * fu * (-dyu * inv_u))
        np.add.at(gy, idx_j, lam * 2.0 * fu * (dyu * inv_u))
        np.add.at(gr, idx_i, lam * 2.0 * fu)
        np.add.at(gr, idx_j, lam * 2.0 * fu)
    out = np.empty(3 * n)
    out[0::3] = gx
    out[1::3] = gy
    out[2::3] = gr
    return out


def _hex_seed(jitter=0.0, rng=None):
    """Generate a hexagonal-ish arrangement scaled to fit, with optional jitter."""
    rng = rng or np.random.default_rng()
    pts = []
    rows = 0
    while len(pts) < N:
        row = rows
        y = row * 0.8660254
        offset = (0.5 if row % 2 else 0.0)
        col = 0
        while len(pts) < N:
            x = col + offset
            pts.append((x, y))
            col += 1
        rows += 1
    pts = np.array(pts[:N])
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1e-9)
    pts = (pts - [xmin, ymin]) / span
    cx = (xmax - xmin) / span
    cy = (ymax - ymin) / span
    pts[:, 0] = 0.5 - cx / 2 + pts[:, 0] * cx
    pts[:, 1] = 0.5 - cy / 2 + pts[:, 1] * cy
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _grid_init(rng):
    """Grid-based initialization with slight jitter."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            x = (i + 0.5) / side + rng.normal(0, 0.02)
            y = (j + 0.5) / side + rng.normal(0, 0.02)
            pts.append([x, y])
    pts = np.array(pts[:N])
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _repulsion_init(rng):
    """Initialize with random positions, then relax via repulsion to spread circles evenly."""
    pts = rng.uniform(0.1, 0.9, size=(N, 2))
    for it in range(300):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pts[i, 0] - pts[j, 0]
                dy = pts[i, 1] - pts[j, 1]
                d = np.hypot(dx, dy)
                min_d = 0.12
                if d < min_d:
                    if d < 1e-8:
                        dx, dy = rng.normal(0, 0.01, 2)
                        d = np.hypot(dx, dy)
                    push = (min_d - d) / d * 0.5
                    pts[i, 0] += dx * push
                    pts[i, 1] += dy * push
                    pts[j, 0] -= dx * push
                    pts[j, 1] -= dy * push
                    moved = True
        pts = np.clip(pts, 0.05, 0.95)
        if not moved:
            break
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _structured_init(rng):
    """Structured initialization: mix of hex, grid, and corner-biased arrangements."""
    # Corner-biased: place circles near corners and edges
    pts = []
    # 4 corners
    for cx, cy in [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]:
        for dx, dy in [(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1)]:
            if len(pts) < N:
                pts.append([cx + dx, cy + dy])
    # Fill remaining with grid-like positions in center
    remaining = N - len(pts)
    for i in range(remaining):
        x = 0.3 + (i % 4) * 0.1 + rng.normal(0, 0.01)
        y = 0.3 + (i // 4) * 0.1 + rng.normal(0, 0.01)
        pts.append([x, y])
    pts = np.array(pts[:N])
    pts = np.clip(pts, 0.05, 0.95)
    rad = np.full(N, 0.0)
    for i in range(N):
        dmin = min(pts[i, 0], 1 - pts[i, 0], pts[i, 1], 1 - pts[i, 1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1]))
        rad[i] = 0.5 * dmin * 0.95
    return pts, rad


def _optimize_from(pos, rad, rng, lams=None, max_iter=300):
    """Optimize from a starting point using annealing schedule."""
    if lams is None:
        lams = [50.0, 20.0, 8.0, 3.0, 1.0, 0.4, 0.15, 0.05]
    bounds = [(0, 1), (0, 1), (1e-4, 1)] * N
    cur_pos, cur_rad = pos.copy(), rad.copy()
    best_val = -np.inf
    best_p, best_r = None, None

    for lam in lams:
        if _remaining() < 8.0:
            break
        z = np.empty(3 * N)
        z[0::3] = cur_pos[:, 0]
        z[1::3] = cur_pos[:, 1]
        z[2::3] = cur_rad
        res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                       bounds=bounds, args=(lam,),
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8})
        z = res.x
        cur_pos = np.column_stack([z[0::3], z[1::3]])
        cur_rad = z[2::3]
        cur_pos, cur_rad, feas = _project_feasible(cur_pos, cur_rad)
        if feas:
            val, _ = _feasible_score(cur_pos, cur_rad)
            if val > best_val:
                best_val = val
                best_p = cur_pos.copy()
                best_r = cur_rad.copy()

    # Grow radii
    if best_p is not None and _remaining() > 10.0:
        p_grown, r_grown = _grow_radii(best_p, best_r, max_iters=50)
        p_grown, r_grown, feas = _project_feasible(p_grown, r_grown, iters=20)
        if feas:
            val, _ = _feasible_score(p_grown, r_grown)
            if val > best_val:
                best_val = val
                best_p = p_grown.copy()
                best_r = r_grown.copy()

    return best_p, best_r, best_val


def construct():
    global _START
    _START = time.time()
    rng = np.random.default_rng(42)

    best_val = -np.inf
    best_pos = None
    best_rad = None

    def _update_best(pos, rad):
        nonlocal best_val, best_pos, best_rad
        val, feas = _feasible_score(pos, rad)
        if feas and val > best_val:
            best_val = val
            best_pos = pos.copy()
            best_rad = rad.copy()

    # Generate diverse seeds
    seeds = []
    # Hex seeds with various jitters
    for jitter in [0.0, 0.02, 0.05, 0.08, 0.12]:
        p, r = _hex_seed(jitter=jitter, rng=rng)
        seeds.append((p, r))
    # Grid seeds
    for _ in range(3):
        p, r = _grid_init(rng)
        seeds.append((p, r))
    # Repulsion seeds
    for _ in range(4):
        p, r = _repulsion_init(rng)
        seeds.append((p, r))
    # Structured seeds
    for _ in range(3):
        p, r = _structured_init(rng)
        seeds.append((p, r))
    # Random seeds
    for _ in range(4):
        p = rng.uniform(0.15, 0.85, size=(N, 2))
        r = rng.uniform(0.05, 0.12, size=N)
        seeds.append((p, r))

    # Phase 1: Multi-start optimization
    for si, (pos, rad) in enumerate(seeds):
        if _remaining() < 12.0:
            break
        p_opt, r_opt, val = _optimize_from(pos, rad, rng, max_iter=250)
        if p_opt is not None:
            _update_best(p_opt, r_opt)

    # Phase 2: Random restarts from best
    if best_pos is not None and _remaining() > 25.0:
        for _ in range(15):
            if _remaining() < 10.0:
                break
            p = best_pos + rng.normal(0, 0.04, size=best_pos.shape)
            p = np.clip(p, 0.05, 0.95)
            r = best_rad * rng.uniform(0.6, 1.0, size=N)
            p_opt, r_opt, val = _optimize_from(p, r, rng, lams=[10.0, 3.0, 0.5, 0.1], max_iter=200)
            if p_opt is not None:
                _update_best(p_opt, r_opt)

    # Phase 3: High-precision final polish
    if best_pos is not None and _remaining() > 15.0:
        for lam in [0.05, 0.02, 0.01, 0.005]:
            if _remaining() < 8.0:
                break
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=[(0, 1), (0, 1), (1e-4, 1)] * N,
                           args=(lam,),
                           options={'maxiter': 500, 'gtol': 1e-9, 'ftol': 1e-12})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=60)
            if feas:
                _update_best(p, r)

        # Final grow
        if _remaining() > 10.0:
            p_grown, r_grown = _grow_radii(best_pos, best_rad, max_iters=100)
            p_grown, r_grown, feas = _project_feasible(p_grown, r_grown, iters=30)
            if feas:
                _update_best(p_grown, r_grown)

        # Another polish after grow
        if _remaining() > 8.0:
            z = np.empty(3 * N)
            z[0::3] = best_pos[:, 0]
            z[1::3] = best_pos[:, 1]
            z[2::3] = best_rad
            res = minimize(_soft_objective, z, jac=_soft_grad, method='L-BFGS-B',
                           bounds=[(0, 1), (0, 1), (1e-4, 1)] * N,
                           args=(0.005,),
                           options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-14})
            z = res.x
            p = np.column_stack([z[0::3], z[1::3]])
            r = z[2::3]
            p, r, feas = _project_feasible(p, r, iters=40)
            if feas:
                p_grown, r_grown = _grow_radii(p, r, max_iters=50)
                p_grown, r_grown, feas = _project_feasible(p_grown, r_grown, iters=20)
                if feas:
                    _update_best(p_grown, r_grown)

    # Fallback
    if best_pos is None:
        p, r = _hex_seed(jitter=0.0)
        p, r, feas = _project_feasible(p, r)
        if feas:
            best_pos, best_rad = p, r
        else:
            side = int(np.ceil(np.sqrt(N)))
            cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                     for i in range(side) for j in range(side)]
            arr = np.array(cells[:N])
            return arr

    out = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return out


if __name__ == '__main__':
    res = construct()
    print("shape", res.shape)
    print("sum r", res[:, 2].sum())
    print("feasible", _feasible_score(res[:, :2], res[:, 2]))
