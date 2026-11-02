import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

T0 = time.time()
DEADLINE = 150.0
SAFETY = 12.0
N = 11

def _hex_verts(cx, cy, a, theta):
    ang = theta + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + a * np.cos(ang), cy + a * np.sin(ang)])

def _edge_axes(verts):
    axes = []
    m = len(verts)
    for i in range(m):
        e = verts[(i + 1) % m] - verts[i]
        n = np.array([-e[1], e[0]])
        n = n / (np.linalg.norm(n) + 1e-18)
        axes.append(n)
    return np.array(axes)

def _sat_gap(A, B):
    axes = np.vstack([_edge_axes(A), _edge_axes(B)])
    best = -np.inf
    for n in axes:
        a = A @ n; b = B @ n
        gap = min(a.max(), b.max()) - max(a.min(), b.min())
        if gap > best:
            best = gap
    return best

def _containment_violation(inner_verts, outer_verts):
    ov = outer_verts
    m = len(ov)
    c = ov.mean(axis=0)
    viol = 0.0
    for i in range(m):
        p1 = ov[i]; p2 = ov[(i + 1) % m]
        e = p2 - p1
        n = np.array([-e[1], e[0]])
        if np.dot(n, c - p1) < 0:
            n = -n
        n = n / (np.linalg.norm(n) + 1e-18)
        d = (inner_verts - p1) @ n
        v = -d.min()
        if v > 0:
            viol += v * v
    return viol

def _decode(params):
    inner = [(params[3*k], params[3*k+1], params[3*k+2]) for k in range(N)]
    ocx, ocy, s, phi = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
    return inner, (ocx, ocy), s, phi

def _encode(inner, center, s, phi):
    p = []
    for (x, y, th) in inner:
        p += [x, y, th]
    p += [center[0], center[1], s, phi]
    return np.array(p)

def _objective(params):
    inner, (ocx, ocy), s, phi = _decode(params)
    s = max(s, 0.5)
    outer = _hex_verts(ocx, ocy, s, phi)
    inners = [_hex_verts(x, y, 1.0, th) for (x, y, th) in inner]
    pen = 0.0
    for iv in inners:
        pen += 1000.0 * _containment_violation(iv, outer)
    for i in range(N):
        for j in range(i + 1, N):
            g = _sat_gap(inners[i], inners[j])
            if g < 0:
                pen += 1000.0 * g * g
    return s + pen

def _is_valid(inner, center, s, phi):
    outer = _hex_verts(center[0], center[1], s, phi)
    inners = [_hex_verts(x, y, 1.0, th) for (x, y, th) in inner]
    for iv in inners:
        if _containment_violation(iv, outer) > 1e-6:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_gap(inners[i], inners[j]) < -1e-6:
                return False
    return True

def construct():
    def initial_guess():
        inner = [(0.0, 0.0, 0.0)]
        d = np.sqrt(3.0)
        for k in range(6):
            a = np.pi / 6 + k * np.pi / 3.0   # flat-normal directions
            inner.append((d * np.cos(a), d * np.sin(a), 0.0))
        d2 = 3.0
        for k in range(4):
            a = k * np.pi / 2.0               # vertex directions
            inner.append((d2 * np.cos(a), d2 * np.sin(a), 0.0))
        return _encode(inner, (0.0, 0.0), 4.5, 0.0)

    p0 = initial_guess()
    best = None
    best_val = np.inf

    def consider(params):
        nonlocal best, best_val
        inner, center, s, phi = _decode(params)
        if s < 0.5:
            return
        if _is_valid(inner, center, s, phi):
            if s < best_val:
                best_val = s
                best = (inner, center, s, phi)

    consider(p0)

    lo = np.empty(3 * N + 4); hi = np.empty(3 * N + 4)
    for k in range(N):
        lo[3*k] = -8; hi[3*k] = 8
        lo[3*k+1] = -8; hi[3*k+1] = 8
        lo[3*k+2] = -np.pi; hi[3*k+2] = np.pi
    lo[3*N] = -4; hi[3*N] = 4
    lo[3*N+1] = -4; hi[3*N+1] = 4
    lo[3*N+2] = 2.0; hi[3*N+2] = 6.0
    lo[3*N+3] = -np.pi/2; hi[3*N+3] = np.pi/2

    def time_ok():
        return (time.time() - T0) < (DEADLINE - SAFETY)

    res = None
    try:
        if time_ok():
            res = differential_evolution(
                _objective, bounds=list(zip(lo, hi)),
                seed=0, maxiter=150, popsize=12, tol=1e-5,
                mutation=(0.3, 1.2), recombination=0.8,
                init='sobol', polish=False, updating='deferred')
            consider(res.x)
    except Exception:
        pass

    starts = []
    if best is not None:
        starts.append(_encode(*best))
    if res is not None:
        starts.append(res.x)
    starts.append(p0)

    for sp in starts:
        if not time_ok():
            break
        try:
            r = minimize(_objective, sp, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7})
            consider(r.x)
        except Exception:
            pass

    if best is None:
        inner = [[np.sqrt(3.0) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner, center, s, phi = best
    inner_out = [[float(x), float(y), float(np.degrees(th)) % 360.0] for (x, y, th) in inner]
    return inner_out, (float(center[0]), float(center[1])), float(s), float(np.degrees(phi) % 360.0)
