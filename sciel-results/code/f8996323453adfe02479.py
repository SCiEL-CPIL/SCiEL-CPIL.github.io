"""Circle packing in unit square: smooth soft-min L-BFGS + contact-graph SLSQP polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26
DEADLINE = 145.0  # safety margin under 150s wall-clock
_START = time.time()


def _elapsed():
    return time.time() - _START


def _valid(c):
    if c is None or c.shape != (N, 3):
        return False
    if not np.all(np.isfinite(c)):
        return False
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r < 1e-10):
        return False
    if np.any(x < -1e-6) or np.any(x > 1 + 1e-6):
        return False
    if np.any(y < -1e-6) or np.any(y > 1 + 1e-6):
        return False
    # wall clearance
    if np.any(r > x + 1e-6) or np.any(r > (1.0 - x) + 1e-6):
        return False
    if np.any(r > y + 1e-6) or np.any(r > (1.0 - y) + 1e-6):
        return False
    # pairwise
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - 1e-6:
                return False
    return True


def _clamp(c):
    x, y, r = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
    x = np.clip(x, 1e-4, 1 - 1e-4)
    y = np.clip(y, 1e-4, 1 - 1e-4)
    r = np.clip(r, 1e-4, 0.5)
    return np.column_stack([x, y, r])


def _score(c):
    if not _valid(c):
        return 0.0
    return float(np.sum(c[:, 2]))


def _objective(v, beta):
    x = v[0:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    # maximize sum r -> minimize -sum r
    obj = -np.sum(r)
    # pairwise soft-min penalty: want dist >= r_i + r_j
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            gap = d - (r[i] + r[j])
            # soft-min of gap: penalize when gap < 0
            # use log-sum-exp of exp(-beta*gap) approx -> penalty = (1/beta)*log(1+exp(-beta*gap))
            if gap < -20.0 / beta:
                pen = -gap
            elif gap > 20.0 / beta:
                pen = 0.0
            else:
                pen = math.log1p(math.exp(-beta * gap)) / beta
            obj += 500.0 * pen
    # wall soft constraints
    for i in range(N):
        for m in (x[i] - r[i], 1.0 - x[i] - r[i], y[i] - r[i], 1.0 - y[i] - r[i]):
            if m < -20.0 / beta:
                pen = -m
            elif m > 20.0 / beta:
                pen = 0.0
            else:
                pen = math.log1p(math.exp(-beta * m)) / beta
            obj += 500.0 * pen
    return obj


def _objective_grad(v, beta):
    x = v[0:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    obj = -np.sum(r)
    gr -= 1.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.hypot(dx, dy)
            gap = d - (r[i] + r[j])
            if gap < -20.0 / beta:
                # hard region: gradient of -gap
                w = 1.0
                if d > 1e-12:
                    gx[i] += 500.0 * w * dx / d
                    gy[i] += 500.0 * w * dy / d
                    gx[j] -= 500.0 * w * dx / d
                    gy[j] -= 500.0 * w * dy / d
                gr[i] += 500.0 * w
                gr[j] += 500.0 * w
                obj += 500.0 * (-gap)
            elif gap > 20.0 / beta:
                pass
            else:
                e = math.exp(-beta * gap)
                pen = math.log1p(e) / beta
                # d pen / d gap = -e/(1+e)
                dp = -e / (1.0 + e)
                if d > 1e-12:
                    gx[i] += 500.0 * dp * dx / d
                    gy[i] += 500.0 * dp * dy / d
                    gx[j] -= 500.0 * dp * dx / d
                    gy[j] -= 500.0 * dp * dy / d
                gr[i] += 500.0 * dp
                gr[j] += 500.0 * dp
                obj += 500.0 * pen
    # walls
    for i in range(N):
        for (axis, sign) in ((0, +1), (0, -1), (1, +1), (1, -1)):
            if axis == 0:
                m = x[i] - r[i] if sign == +1 else 1.0 - x[i] - r[i]
            else:
                m = y[i] - r[i] if sign == +1 else 1.0 - y[i] - r[i]
            if m < -20.0 / beta:
                w = 1.0
                if axis == 0:
                    gx[i] += 500.0 * w * sign
                else:
                    gy[i] += 500.0 * w * sign
                gr[i] += 500.0 * w * (-sign)
                obj += 500.0 * (-m)
            elif m > 20.0 / beta:
                pass
            else:
                e = math.exp(-beta * m)
                pen = math.log1p(e) / beta
                dp = -e / (1.0 + e)
                if axis == 0:
                    gx[i] += 500.0 * dp * sign
                else:
                    gy[i] += 500.0 * dp * sign
                gr[i] += 500.0 * dp * (-sign)
                obj += 500.0 * pen
    return obj, np.concatenate([gx, gy, gr])


def _slsqp_polish(c, iters=600, tol=1e-9):
    """SLSQP on active contact constraints to maximize sum r."""
    c = _clamp(c.copy())
    x, y, r = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
    best = c.copy()
    best_val = _score(best)

    def obj(v):
        return -np.sum(v[2 * N:])

    def obj_grad(v):
        g = np.zeros(3 * N)
        g[2 * N:] = -1.0
        return g

    # active pairs: those close to touching
    def active_pairs():
        act = []
        for i in range(N):
            for j in range(i + 1, N):
                d = math.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j] + 0.05 * (r[i] + r[j]):
                    act.append((i, j))
        return act

    def wall_active():
        wa = []
        for i in range(N):
            if x[i] - r[i] < 0.05 * r[i]:
                wa.append((i, 'L'))
            if 1.0 - x[i] - r[i] < 0.05 * r[i]:
                wa.append((i, 'R'))
            if y[i] - r[i] < 0.05 * r[i]:
                wa.append((i, 'B'))
            if 1.0 - y[i] - r[i] < 0.05 * r[i]:
                wa.append((i, 'T'))
        return wa

    # Build LinearConstraint for active pairs and walls
    def build_constraints():
        cons = []
        for (i, j) in active_pairs():
            A = np.zeros((1, 3 * N))
            A[0, i] = 1.0
            A[0, N + i] = 0.0
            A[0, 2 * N + i] = -1.0
            A[0, j] = -1.0
            A[0, N + j] = 0.0
            A[0, 2 * N + j] = -1.0
            b = np.array([0.0])
            cons.append(LinearConstraint(A, b, None))
        for (i, w) in wall_active():
            A = np.zeros((1, 3 * N))
            if w == 'L':
                A[0, i] = 1.0
                A[0, 2 * N + i] = -1.0
            elif w == 'R':
                A[0, i] = -1.0
                A[0, 2 * N + i] = -1.0
            elif w == 'B':
                A[0, N + i] = 1.0
                A[0, 2 * N + i] = -1.0
            elif w == 'T':
                A[0, N + i] = -1.0
                A[0, 2 * N + i] = -1.0
            b = np.array([0.0])
            cons.append(LinearConstraint(A, b, None))
        return cons

    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.5)] * N
    v = c.flatten()
    for it in range(iters):
        if _elapsed() > DEADLINE:
            break
        cons = build_constraints()
        try:
            res = minimize(obj, v, jac=obj_grad, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 50, 'ftol': tol})
            if res.success or res.fun < obj(v):
                v = np.clip(res.x, 0, 1)
                v[2 * N:] = np.clip(v[2 * N:], 1e-4, 0.5)
                c = _clamp(v.reshape(N, 3))
                val = _score(c)
                if val > best_val:
                    best_val = val
                    best = c.copy()
        except Exception:
            pass
    return best


def _grid_start(jitter=0.0):
    side = int(math.ceil(math.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            if jitter > 0:
                cx += (np.random.rand() - 0.5) * jitter / side
                cy += (np.random.rand() - 0.5) * jitter / side
            r = 0.5 / side * 0.98
            cells.append((cx, cy, r))
    return np.array(cells[:N])


def _random_start(n_try=2000):
    # place circles with repulsion, greedy
    rng = np.random.default_rng(12345)
    pts = []
    for _ in range(n_try):
        x = rng.random()
        y = rng.random()
        # estimate max radius by distance to walls and existing
        d = min(x, 1 - x, y, 1 - y)
        for (px, py, pr) in pts:
            d = min(d, math.hypot(x - px, y - py) - pr)
        if d > 0.02:
            pts.append((x, y, d * 0.95))
        if len(pts) >= N:
            break
    if len(pts) < N:
        return None
    # take largest N
    pts.sort(key=lambda p: -p[2])
    return np.array(pts[:N])


def _voronoi_start():
    # use scipy Voronoi on random points, radius = min distance to edges/neighbors
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(777)
    pts = rng.random((N, 2)) * 0.8 + 0.1
    try:
        vor = Voronoi(pts)
    except Exception:
        return None
    radii = []
    for i, (px, py) in enumerate(pts):
        d = min(px, 1 - px, py, 1 - py)
        for ridge in vor.ridge_points[i]:
            if ridge < 0:
                continue
            j = vor.ridge_vertices[i][ridge]
            if j < 0:
                continue
            vx, vy = vor.vertices[j]
            d = min(d, math.hypot(px - vx, py - vy))
        radii.append(d * 0.95)
    return np.column_stack([pts, np.array(radii)])


def construct():
    best = None
    best_val = 0.0

    candidates = []
    g = _grid_start(0.0)
    if _valid(g):
        candidates.append(g)
    for jit in (0.1, 0.2, 0.3):
        gg = _grid_start(jit)
        if _valid(gg):
            candidates.append(gg)
    rr = _random_start()
    if rr is not None and _valid(rr):
        candidates.append(rr)
    vv = _voronoi_start()
    if vv is not None and _valid(vv):
        candidates.append(vv)

    for c in candidates:
        if _elapsed() > DEADLINE:
            break
        # smooth L-BFGS refinement with increasing beta
        v = c.flatten()
        for beta in (10.0, 30.0, 100.0, 300.0):
            if _elapsed() > DEADLINE:
                break
            try:
                res = minimize(_objective, v, jac=_objective_grad, args=(beta,),
                               method='L-BFGS-B',
                               bounds=[(0.0, 1.0)] * (2 * N) + [(1e-4, 0.5)] * N,
                               options={'maxiter': 200, 'ftol': 1e-12})
                v = res.x
            except Exception:
                pass
        c2 = _clamp(v.reshape(N, 3))
        if _valid(c2):
            c2 = _slsqp_polish(c2)
            val = _score(c2)
            if val > best_val:
                best_val = val
                best = c2

    # final safety: ensure we return something valid
    if best is None or not _valid(best):
        # fallback grid
        side = int(math.ceil(math.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    return best
