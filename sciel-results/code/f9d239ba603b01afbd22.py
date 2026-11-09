# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = time.time() + 560.0  # safety margin before 630s kill
_TIME0 = time.time()


def _elapsed():
    return time.time() - _TIME0


def _dead():
    return time.time() >= _DEADLINE


def _base_inside(u, v):
    """True if local point (u,v) is in the base L-corridor (corner at origin, a=0)."""
    # strip 1: 0 <= u <= 1, -3 <= v <= 1
    # strip 2: -3 <= u <= 1, 0 <= v <= 1
    return ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
           ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))


def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y


def _eval_path(X, Y, path):
    """Count grid points inside every placement in `path` (array of (x,y,a_deg))."""
    n = X.shape[0]
    m = X.shape[1]
    # work in flat arrays for speed
    xf = X.ravel()
    yf = Y.ravel()
    total = xf.size
    # start with all True
    inside = np.ones(total, dtype=bool)
    for (cx, cy, adeg) in path:
        a = math.radians(adeg)
        c = math.cos(a)
        s = math.sin(a)
        # local = R(-a) * (w - (cx,cy)); R(-a) = [[c, s],[-s, c]]
        dx = xf - cx
        dy = yf - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        ins = _base_inside(u, v)
        inside &= ins
        if not inside.any():
            break
    return int(inside.sum())


def _full_path(path):
    """Prepend start, append end, and densify by angle."""
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    pts = [start] + list(path) + [end]
    densified = [pts[0]]
    for k in range(1, len(pts)):
        p1 = pts[k - 1]
        p2 = pts[k]
        da = abs(p2[2] - p1[2])
        m = max(1, int(math.ceil(da)))
        for t in range(1, m + 1):
            f = t / m
            densified.append(p1 + f * (p2 - p1))
    return np.array(densified)


def _score(path):
    """Score of a raw path (without start/end, which are added by evaluator)."""
    fp = _full_path(path)
    return _eval_path(_GX, _GY, fp)


# Precompute grid once
_GX, _GY = _build_grid()


def _initial_path():
    # Hand-tuned starting path: slide corner along an arc while rotating 0->90.
    # We want the corridor to "sweep" so the intersection stays large.
    # Start (0,0,0) is fixed by evaluator; end (-2,0,90) fixed.
    # Intermediate: move corner so that at each angle the corridor covers the sofa.
    # A reasonable arc: corner traces a curve from near (0,0) to (-2,0).
    ang = np.linspace(10.0, 80.0, 8)
    # radius-like placement: put corner at distance so strips overlap the sofa region.
    path = []
    for a in ang:
        ar = math.radians(a)
        # Place inner corner along an arc of radius ~1.1 centered near the turn.
        # The sofa lives roughly in the square [-1,1]x[-1,1] near the corner region.
        # We push the corner so that the two strips of the corridor cover the sofa.
        # Use a parametric arc: x = -1.0 - 0.5*cos?  Let's use a smooth curve.
        x = -1.0 * (1.0 - math.cos(ar)) - 0.5 * math.sin(ar)
        y = 0.6 * math.sin(ar) - 0.2 * (1.0 - math.cos(ar))
        path.append([x, y, float(a)])
    return np.array(path, dtype=np.float64)


def _nelder_mead(f, x0, step=0.15, max_iter=400, tol=1e-9):
    """Simple Nelder-Mead on a vector. f returns scalar (we maximize)."""
    n = len(x0)
    simplex = [x0.copy() for _ in range(n + 1)]
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += step
    fv = [f(s) for s in simplex]
    best = 0
    for it in range(max_iter):
        if _dead():
            break
        order = np.argsort(fv)  # ascending (we maximize, so worst = index 0)
        simplex = [simplex[i] for i in order]
        fv = [fv[i] for i in order]
        if abs(fv[-1] - fv[0]) < tol:
            break
        worst = simplex[0]
        # centroid of all but worst
        centroid = np.mean(simplex[1:], axis=0)
        # reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - worst)
        fr = f(xr)
        if fv[1] < fr <= fv[-1]:
            simplex[0] = xr
            fv[0] = fr
            continue
        if fr > fv[-1]:
            # expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - worst)
            fe = f(xe)
            if fe > fr:
                simplex[0] = xe
                fv[0] = fe
            else:
                simplex[0] = xr
                fv[0] = fr
            continue
        # contraction
        rho = 0.5
        if fr < fv[1]:
            # outside contraction
            xc = centroid - rho * (centroid - worst)
        else:
            # inside contraction
            xc = centroid + rho * (centroid - worst)
        fc = f(xc)
        if fv[1] <= fc < fv[-1]:
            simplex[0] = xc
            fv[0] = fc
            continue
        # shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            fv[i] = f(simplex[i])
    best_idx = int(np.argmax(fv))
    return simplex[best_idx], fv[best_idx]


def _coordinate_refine(x, f, step=0.05, rounds=6):
    """Coordinate descent with shrinking step to fine-tune each (x,y,a)."""
    cur = x.copy()
    curf = f(cur)
    for r in range(rounds):
        if _dead():
            break
        improved = False
        for i in range(len(cur)):
            for delta in (step, -step):
                trial = cur.copy()
                trial[i] += delta
                tf = f(trial)
                if tf > curf:
                    cur = trial
                    curf = tf
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break
    return cur, curf


def moving_sofa_2d() -> np.ndarray:
    global _GX, _GY
    if _GX is None:
        _GX, _GY = _build_grid()

    def f_vec(v):
        # v is flat length 3*n; reshape to (n,3)
        n = len(v) // 3
        path = v.reshape(n, 3)
        return _score(path)

    # Start from a decent initial path
    path0 = _initial_path()
    v0 = path0.ravel().copy()

    best_v = v0.copy()
    best_f = f_vec(best_v)

    # Nelder-Mead on the whole vector
    if not _dead():
        v1, f1 = _nelder_mead(f_vec, v0, step=0.2, max_iter=300)
        if f1 > best_f:
            best_v, best_f = v1, f1

    # Coordinate refinement
    if not _dead():
        v2, f2 = _coordinate_refine(best_v, f_vec, step=0.05, rounds=8)
        if f2 > best_f:
            best_v, best_f = v2, f2

    # A second NM pass from the refined point with smaller step
    if not _dead():
        v3, f3 = _nelder_mead(f_vec, best_v, step=0.08, max_iter=200)
        if f3 > best_f:
            best_v, best_f = v3, f3
        if not _dead():
            v4, f4 = _coordinate_refine(best_v, f_vec, step=0.03, rounds=6)
            if f4 > best_f:
                best_v, best_f = v4, f4

    n = len(best_v) // 3
    path = best_v.reshape(n, 3).astype(np.float64)
    # Ensure finiteness
    if not np.all(np.isfinite(path)):
        path = _initial_path()
    return path


# EVOLVE-BLOCK-END
