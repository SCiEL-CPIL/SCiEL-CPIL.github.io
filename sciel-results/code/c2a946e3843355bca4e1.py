"""Packing 12 unit hexagons into a minimal enclosing regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 12
DEADLINE = time.time() + 140.0
EPS = 1e-9

# Vertices of a unit regular hexagon (circumradius = 1) for angle 0
_HX0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                 for a in range(0, 360, 60)])
_N6 = len(_HX0)
_NN = N * _N6


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _verts(cx, cy, ang):
    return (cx, cy) + _HX0 @ _rot(ang).T


def _min_hex_lp(pos, ang, oang):
    """Min enclosing regular hexagon (center free, orientation oang) via LP."""
    R = _rot(oang)
    # A = [R, -I] (2 x 4) ; c = [cx, cy, 1, 0]
    A = np.hstack([R, -np.eye(2)])
    c = np.array([0.0, 0.0, 1.0, 0.0])
    Ab = np.zeros((_NN, 4))
    cb = np.zeros(_NN)
    k = 0
    for i in range(N):
        v = _verts(pos[i, 0], pos[i, 1], ang[i])
        for j in range(_N6):
            Ab[k] = A @ v[j]
            cb[k] = c @ v[j]
            k += 1
    res = linprog(np.zeros(4), A_ub=Ab, b_ub=cb,
                  bounds=[(None, None), (None, None), (0.0, None), (0.0, None)],
                  method='highs')
    if not res.success:
        return None
    return float(res.x[2])


def _outer_vertices(s, oang, cx, cy):
    return np.array([[s * math.cos(math.radians(a + oang)),
                      s * math.sin(math.radians(a + oang))] for a in range(0, 360, 60)]) + (cx, cy)


def _inner_vertices(pos, ang):
    out = np.zeros((N, _N6, 2))
    for i in range(N):
        out[i] = _verts(pos[i, 0], pos[i, 1], ang[i])
    return out


def _check_fits(pos, ang, oang, s, cx, cy):
    ovs = _outer_vertices(s, oang, cx, cy)
    ovs = np.vstack([ovs, ovs[0]])
    oseg = np.stack([ovs[:-1], ovs[1:]], axis=1)
    ivs = _inner_vertices(pos, ang)
    for i in range(N):
        for j in range(_N6):
            p = ivs[i, j]
            ok = False
            for (a, b) in oseg:
                ab = b - a
                t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
                t = max(0.0, min(1.0, t))
                q = a + t * ab
                if np.dot(p - q, p - q) <= 1e-6 * (s * s + 1.0):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _lattice_init(theta, cx, cy):
    """Positions/angles of a hexagonal lattice clipped to a radius, centered at (cx,cy)."""
    R = _rot(theta)
    # axial coords
    coords = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = q + 0.5 * r
            y = math.sqrt(3) / 2.0 * r
            coords.append((x, y))
    pts = np.array([(R @ np.array([x, y])).tolist() for (x, y) in coords])
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    sel = order[:N]
    pos = pts[sel] + np.array([cx, cy])
    ang = np.full(N, theta)
    return pos, ang


def _opt_inner(pos0, ang0, oang, s0, cx, cy, tlim):
    def obj(v):
        if time.time() > tlim:
            return 1e8
        p = v[:2 * N].reshape(N, 2)
        a = v[2 * N:]
        r = _min_hex_lp(p, a, oang)
        return r if r is not None else 1e8
    v0 = np.concatenate([pos0.ravel(), ang0])
    res = minimize(obj, v0, method='SLSQP',
                   options={'maxiter': 400, 'ftol': 1e-10, 'disp': False})
    if res.success or (hasattr(res, 'fun') and res.fun < 1e7):
        p = res.x[:2 * N].reshape(N, 2)
        a = res.x[2 * N:]
        r = _min_hex_lp(p, a, oang)
        if r is not None and r < 1e7:
            return p, a, r
    return pos0, ang0, s0


def construct():
    t0 = time.time()
    best = (40.0, None, None, None, None)  # (s, pos, ang, oang, center)

    for theta in [0.0, math.radians(10), math.radians(30), math.radians(60),
                  math.radians(90), math.radians(120)]:
        if time.time() > DEADLINE:
            break
        pos, ang = _lattice_init(theta, 0.0, 0.0)
        s0 = _min_hex_lp(pos, ang, theta)
        if s0 is None:
            continue
        if s0 < best[0]:
            best = (s0, pos.copy(), ang.copy(), theta, (0.0, 0.0))
        if time.time() > DEADLINE:
            break
        # polish inner positions/angles
        p, a, s = _opt_inner(pos, ang, theta, s0, 0.0, 0.0, DEADLINE)
        if s < best[0]:
            best = (s, p, a, theta, (0.0, 0.0))

    s, pos, ang, oang, center = best
    # Final coarse sweep over outer orientation with the best inner layout
    for oa in np.linspace(0.0, math.radians(60), 13):
        if time.time() > DEADLINE:
            break
        r = _min_hex_lp(pos, ang, oa)
        if r is not None and r < best[0]:
            best = (r, pos, ang, oa, (0.0, 0.0))
    s, pos, ang, oang, center = best

    # Safety: if layout doesn't fit the LP value (numerical), bump side length
    if not _check_fits(pos, ang, oang, s, center[0], center[1]):
        s = s * 1.02 + 0.05

    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (float(center[0]), float(center[1])), float(s), float(math.degrees(oang))


if __name__ == '__main__':
    inner, center, side, angle = construct()
    print('side =', side)
