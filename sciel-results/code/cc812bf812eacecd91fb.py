"""Circle packing in unit square: PBD + Delaunay polish."""
import time
import numpy as np
from scipy.spatial import Delaunay, distance_matrix

N = 26
_DEADLINE = 140.0  # wall-clock safety margin (limit is 150s)


def _valid(circles):
    """Check circles are inside unit square with disjoint interiors."""
    if circles is None:
        return False
    c = np.asarray(circles, dtype=float)
    if c.ndim != 2 or c.shape[0] != N or c.shape[1] != 3:
        return False
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # pairwise disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-7:
                return False
    return True


def _score(circles):
    if not _valid(circles):
        return -np.inf
    return float(np.sum(circles[:, 2]))


def _hex_init():
    """Dense hexagonal lattice init with small radii."""
    # Hex lattice: rows offset. Use enough points, take N.
    pts = []
    s = 0.18  # spacing
    for j in range(0, 12):
        row_y = (j + 0.5) * s
        offset = (s / 2.0) * (j % 2)
        for i in range(0, 12):
            x = (i + 0.5) * s + offset
            y = row_y
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    # if not enough, fallback to grid
    if len(pts) < N:
        side = int(np.ceil(np.sqrt(N)))
        pts = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    pts = pts[:N]
    r0 = 0.04
    return np.array([[x, y, r0] for (x, y) in pts])


def _pbd_step(circles, lam, t):
    """One PBD iteration: apply pressure (grow radii), then Gauss-Seidel constraints."""
    c = circles.copy()
    x = c[:, 0].copy()
    y = c[:, 1].copy()
    r = c[:, 2].copy()

    # Pressure: grow radii (global outward force)
    r = r + lam * t * 0.001

    # Gauss-Seidel constraint solving (several passes for stability)
    n = len(x)
    for _pass in range(3):
        # Pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                if d < min_d - 1e-12:
                    if d < 1e-12:
                        dx, dy = 1e-6, 0.0
                        d = 1e-6
                    # equal mass split
                    corr = (min_d - d) / (2.0 * d)
                    x[i] += dx * corr
                    y[i] += dy * corr
                    x[j] -= dx * corr
                    y[j] -= dy * corr
        # Boundary containment
        for i in range(n):
            if x[i] - r[i] < 0:
                x[i] = r[i]
            if x[i] + r[i] > 1:
                x[i] = 1 - r[i]
            if y[i] - r[i] < 0:
                y[i] = r[i]
            if y[i] + r[i] > 1:
                y[i] = 1 - r[i]
    return np.column_stack([x, y, r])


def _delaunay_polish(circles, max_iter=4000, tol=1e-9, lr0=0.02):
    """Local polish: maximize sum radii subject to non-overlap + boundary via gradient/penalty."""
    c = circles.copy().astype(float)
    x = c[:, 0]
    y = c[:, 1]
    r = c[:, 2]
    n = len(x)
    best = c.copy()
    best_s = _score(c)

    # Build contact graph from Delaunay of centers
    try:
        tri = Delaunay(np.column_stack([x, y]))
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    edges.add((int(simplex[a]), int(simplex[b])))
        edges = list(edges)
    except Exception:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Penalty weights for constraints
    w_pen = 500.0
    lr = lr0

    for it in range(max_iter):
        # Compute gradient of objective - penalty
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)
        # objective: +r
        gr[:] = 1.0
        # penalty for overlap: for each edge, if d < r_i+r_j, push apart
        for (i, j) in edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            gap = r[i] + r[j] - d
            if gap > 0:
                if d < 1e-12:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / d, dy / d
                # penalty gradient: increase d (separate) and reduce radii
                gx[i] -= w_pen * ux
                gy[i] -= w_pen * uy
                gx[j] += w_pen * ux
                gy[j] += w_pen * uy
                gr[i] -= w_pen * 0.5
                gr[j] -= w_pen * 0.5
        # boundary penalty
        for i in range(n):
            if x[i] - r[i] < 0:
                gx[i] += w_pen
                gr[i] -= w_pen
            if x[i] + r[i] > 1:
                gx[i] -= w_pen
                gr[i] -= w_pen
            if y[i] - r[i] < 0:
                gy[i] += w_pen
                gr[i] -= w_pen
            if y[i] + r[i] > 1:
                gy[i] -= w_pen
                gr[i] -= w_pen

        # line search: try step, accept if valid & improves
        step = lr
        improved = False
        for _ls in range(6):
            nx = x + step * gx
            ny = y + step * gy
            nr = r + step * gr
            nr = np.clip(nr, 1e-4, 0.5)
            cand = np.column_stack([nx, ny, nr])
            s = _score(cand)
            if np.isfinite(s) and s > best_s + 1e-12:
                x, y, r = nx, ny, nr
                best = cand.copy()
                best_s = s
                improved = True
                break
            step *= 0.5
        if not improved:
            lr *= 0.85
            if lr < 1e-6:
                break
        else:
            # slow decay
            lr = min(lr0, lr * 1.001)
    return best, best_s


def construct():
    t0 = time.time()
    best = _hex_init()
    best_s = _score(best)

    # PBD phase: grow radii while resolving constraints
    lam = 1.0
    t = 1.0
    it = 0
    max_pbd = 6000
    while it < max_pbd:
        if time.time() > _DEADLINE - 30:
            break
        c = _pbd_step(best, lam, t)
        s = _score(c)
        if np.isfinite(s) and s > best_s:
            best_s = s
            best = c
        it += 1
        # adaptive: decay pressure as we approach limit
        t = max(0.05, t * 0.9995)

    # Delaunay polish
    if time.time() < _DEADLINE - 20:
        polished, s = _delaunay_polish(best)
        if np.isfinite(s) and s > best_s:
            best = polished
            best_s = s

    # Final high-precision polish if time remains
    if time.time() < _DEADLINE - 10:
        polished, s = _delaunay_polish(best, max_iter=2000, tol=1e-10, lr0=0.005)
        if np.isfinite(s) and s > best_s:
            best = polished
            best_s = s

    # Ensure validity; fallback to valid seed if needed
    if not _valid(best):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    return best
