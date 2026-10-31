"""Packing 11 unit hexagons inside a regular hexagon.

Strategy: start from a dense hexagonal (honeycomb) cluster of 11 unit hexagons
and refine the outer side length, inner translations and rotations with a
gradient-free optimizer to minimize the outer side length while keeping the
inner hexagons disjoint and fully inside the outer hexagon.
"""
import time
import math
import numpy as np

N = 11
EPS = 1e-6          # small epsilon to avoid floating-point false overlaps
DEADLINE = 130.0    # wall-clock safety margin (evaluator kills at ~150 s)


def _hex_verts(cx, cy, r, ang):
    """Vertices of a regular hexagon (circumradius r) centred at (cx, cy),
    rotated by ang radians.  Vertex k is at angle ang + k*pi/3."""
    a = ang + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def _signed_dist_hexagon(p, R, theta):
    """Signed distance from point p to a regular hexagon of circumradius R
    rotated by theta.  Positive outside, negative inside.  The hexagon's six
    sides lie at distance a = R*cos(pi/6) from its centre, with outward
    normals at angles theta + pi/6 + k*pi/3."""
    a = R * math.cos(math.pi / 6.0)
    n = theta + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    d = (p[:, 0] * np.cos(n) + p[:, 1] * np.sin(n)) - a
    return d.max()


def _min_gap(v1, v2):
    """Minimum separation between two convex polygons (each >= 3 vertices).
    Returns (distance, (point_on_v1, point_on_v2))."""
    best = math.inf
    bp = (None, None)
    for i in range(len(v1)):
        a = v1[i]; b = v1[(i + 1) % len(v1)]
        e1 = b - a
        for j in range(len(v2)):
            c = v2[j]; d = v2[(j + 1) % len(v2)]
            e2 = d - c
            # point-segment distances (both directions)
            for p, q, seg in ((a, c, e1), (b, c, e1), (c, a, e2), (d, a, e2)):
                t = np.dot(p - q, seg) / max(np.dot(seg, seg), 1e-12)
                t = min(max(t, 0.0), 1.0)
                proj = q + t * seg
                dist = np.linalg.norm(p - proj)
                if dist < best:
                    best = dist
                    bp = (p, proj)
            # segment-segment intersection
            denom = e1[0] * e2[1] - e1[1] * e2[0]
            if abs(denom) > 1e-12:
                t = ((c - a)[0] * e2[1] - (c - a)[1] * e2[0]) / denom
                s = ((c - a)[0] * e1[1] - (c - a)[1] * e1[0]) / denom
                if -EPS <= t <= 1 + EPS and -EPS <= s <= 1 + EPS:
                    return 0.0, (a + t * e1, c + s * e2)
    return best, bp


def _cluster_positions():
    """Hexagonal cluster of 11 unit hexagons on a triangular lattice
    (spacing = sqrt(3), i.e. flat-to-flat touching)."""
    s = math.sqrt(3.0)
    # axial coordinates (q, r)
    axial = [(0, 0)]
    for k in range(1, 3):
        for q in range(-k, k + 1):
            for r in range(max(-k, -q - k), min(k, -q + k) + 1):
                axial.append((q, r))
    axial = axial[:N]
    pts = []
    for q, r in axial:
        x = s * (q + r / 2.0)
        y = s * (math.sqrt(3) / 2.0) * r
        pts.append((x, y))
    return pts


def construct():
    t0 = time.time()
    pts = _cluster_positions()
    cx0 = float(np.mean([p[0] for p in pts]))
    cy0 = float(np.mean([p[1] for p in pts]))
    pts = [(x - cx0, y - cy0) for (x, y) in pts]

    # initial outer hexagon sized to contain the cluster with a small margin
    def _required_R(th):
        m = 0.0
        for (x, y) in pts:
            m = max(m, _signed_dist_hexagon(np.array([[x, y]]), 1.0, th) + 1.0)
        return max(m, 1.0)

    R = _required_R(0.0) + 0.2

    # optimisation state: [R, cx_i, cy_i, ang_i ...]
    x = np.array([R] + [v for p in pts for v in (p[0], p[1], 0.0)])

    def _verts_of(xv, i):
        cx = xv[1 + 3 * i]; cy = xv[2 + 3 * i]; ang = xv[3 + 3 * i]
        return _hex_verts(cx, cy, 1.0, ang)

    def _score(xv):
        """Objective = R + penalty.  Lower is better."""
        R = xv[0]
        outer = _hex_verts(0.0, 0.0, R, 0.0)
        pen = 0.0
        verts = [_verts_of(xv, i) for i in range(N)]
        # containment
        for i in range(N):
            d = _signed_dist_hexagon(verts[i], R, 0.0).max()
            if d > 0.0:
                pen += 1e4 * d + 1e2 * d * d
        # pairwise disjointness
        for i in range(N):
            for j in range(i + 1, N):
                gap, _ = _min_gap(verts[i], verts[j])
                if gap < EPS:
                    pen += 1e4 * (EPS - gap) + 1e2 * (EPS - gap) ** 2
        return R + pen

    best_x = x.copy()
    best_s = _score(x)

    # coordinate descent with golden-section line search
    def _line_search(xv, d, f0):
        g1 = (math.sqrt(5) - 1) / 2.0
        a, b = 0.0, 1.0
        xb = b
        fb = f0
        xs = a + (b - a) * g1
        fs = _score(xv + xs * d)
        it = 0
        while b - a > 1e-9 and it < 40:
            it += 1
            if fs < fb:
                b, xb, fb = xs, xb, fs
                xs = a + (b - a) * g1
                fs = _score(xv + xs * d)
            else:
                a, xs = xs, b - (b - a) * g1
                fs = _score(xv + xs * d)
        return _score(xv + xb * d), xb

    step_scale = {0: 0.5, 1: 0.1, 2: 0.1, 3: 0.1}
    for _pass in range(40):
        if time.time() - t0 > DEADLINE:
            break
        improved = False
        for k in range(len(x)):
            if time.time() - t0 > DEADLINE:
                break
            d = np.zeros_like(x)
            d[k] = step_scale.get(k, 0.1)
            f0 = _score(x)
            fs, xb = _line_search(x, d, f0)
            if fs < f0 - 1e-12:
                x = x + xb * d
                if fs < best_s:
                    best_s = fs
                    best_x = x.copy()
                improved = True
        if not improved:
            # small random kicks to escape local minima
            kicked = False
            for _ in range(6):
                if time.time() - t0 > DEADLINE:
                    break
                kk = np.random.randint(1, len(x))
                d = np.zeros_like(x)
                d[kk] = 0.05 * (1 if np.random.random() < 0.5 else -1)
                f0 = _score(x)
                fs, xb = _line_search(x, d, f0)
                if fs < f0 - 1e-12:
                    x = x + xb * d
                    if fs < best_s:
                        best_s = fs
                        best_x = x.copy()
                    kicked = True
            if not kicked:
                break

    # final polish: reduce R as much as possible while staying feasible
    xv = best_x.copy()
    for _ in range(60):
        if time.time() - t0 > DEADLINE:
            break
        R = xv[0]
        d = np.zeros_like(xv)
        d[0] = -0.05
        f0 = _score(xv)
        fs, xb = _line_search(xv, d, f0)
        if fs < f0 - 1e-12 and fs < best_s:
            xv = xv + xb * d
            best_s = fs
            best_x = xv.copy()
        else:
            break

    R = float(best_x[0])
    inner = []
    for i in range(N):
        cx = float(best_x[1 + 3 * i])
        cy = float(best_x[2 + 3 * i])
        ang = float(best_x[3 + 3 * i])
        inner.append([cx, cy, math.degrees(ang)])
    return inner, (0.0, 0.0), R, 0.0
