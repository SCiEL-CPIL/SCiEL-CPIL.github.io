"""Pack N unit regular hexagons inside a regular hexagon; minimize outer side length."""
import time
import math
import random
import numpy as np

N = 11
SEED = 12345
T_START = time.time()
TIME_LIMIT = 115.0          # safety margin before the 120s wall-clock kill

# ---- geometry helpers -------------------------------------------------------
def _hex_vertices(cx, cy, s, ang):
    """Vertices of a regular hexagon of side s, center (cx,cy), rotation ang (rad)."""
    a = np.deg2rad(ang)
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _halfspace_dist(p, a, b):
    """Signed distance of point p to line a->b (positive on the left of a->b)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return 0.0
    nx = -dy / L
    ny = dx / L
    return nx * (p[0] - a[0]) + ny * (p[1] - a[1])

def _innermost_point(P):
    """Point of polygon P (convex, CCW) farthest from the origin."""
    d = P[:, 0] ** 2 + P[:, 1] ** 2
    return P[int(np.argmax(d))]

def _is_inside_hex(p, P):
    """Is point p inside (or on) convex CCW polygon P?"""
    for i in range(len(P)):
        if _halfspace_dist(p, P[i], P[(i + 1) % len(P)]) < -1e-9:
            return False
    return True

def _segment_interior_intersect(P, Q):
    """Do edges of convex polygons P,Q cross strictly in their interiors?"""
    n, m = len(P), len(Q)
    for i in range(n):
        a1, a2 = P[i], P[(i + 1) % n]
        for j in range(m):
            b1, b2 = Q[j], Q[(j + 1) % m]
            d1 = _halfspace_dist(b1, a1, a2)
            d2 = _halfspace_dist(b2, a1, a2)
            d3 = _halfspace_dist(a1, b1, b2)
            d4 = _halfspace_dist(a2, b1, b2)
            if (d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9):
                if (d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9):
                    return True
    return False

def _overlap(P, Q):
    """True if convex polygons P,Q have a non-negligible interior intersection."""
    if _segment_interior_intersect(P, Q):
        return True
    ip = _innermost_point(P)
    if _is_inside_hex(ip, Q):
        return True
    iq = _innermost_point(Q)
    if _is_inside_hex(iq, P):
        return True
    return False

def _polygons(x, y, ang, s):
    """List of vertex arrays for the N hexagons."""
    out = []
    for k in range(N):
        out.append(_hex_vertices(x[k], y[k], s, ang[k]))
    return out

def _feasible(x, y, ang, s, S):
    """True if all N unit hexagons are disjoint and inside the outer hexagon side S."""
    outer = _hex_vertices(0.0, 0.0, S, 0.0)
    polys = _polygons(x, y, ang, s)
    for P in polys:
        if not _is_inside_hex(_innermost_point(P), outer):
            return False
    for i in range(N):
        for j in range(i + 1, N):
            if _overlap(polys[i], polys[j]):
                return False
    return True

# ---- multi-start local optimizer -------------------------------------------
def _optimize(x, y, ang, s, S, iters):
    """Hill-climb: shrink S while fixing feasibility via small nudges."""
    best = (S, x.copy(), y.copy(), ang.copy())
    rng = random.Random()
    for it in range(iters):
        if time.time() - T_START > TIME_LIMIT - 5:
            break
        # try to shrink the container
        S_try = S * 0.995
        if S_try >= 1.0 and _feasible(x, y, ang, s, S_try):
            S = S_try
            best = (S, x.copy(), y.copy(), ang.copy())
            continue
        # otherwise nudge the most "offending" hexagon and keep S
        moved = False
        for _ in range(30):
            k = rng.randrange(N)
            step = rng.uniform(0.05, 0.4)
            dirx = rng.uniform(-1.0, 1.0)
            diry = rng.uniform(-1.0, 1.0)
            d = math.hypot(dirx, diry) or 1.0
            nx = x[k] + step * dirx / d
            ny = y[k] + step * diry / d
            na = ang[k] + rng.uniform(-0.15, 0.15)
            if _feasible(np.concatenate([x[:k], [nx], x[k + 1:]]),
                         np.concatenate([y[:k], [ny], y[k + 1:]]),
                         np.concatenate([ang[:k], [na], ang[k + 1:]]),
                         s, S):
                x[k], y[k], ang[k] = nx, ny, na
                moved = True
                break
        if not moved:
            # small random restart of a single hexagon to escape
            k = rng.randrange(N)
            x[k] = rng.uniform(-S * 0.5, S * 0.5)
            y[k] = rng.uniform(-S * 0.5, S * 0.5)
            ang[k] = rng.uniform(0.0, 60.0)
    return best

# ---- initial layouts --------------------------------------------------------
def _hex_grid_layout():
    """Honeycomb packing of N hexagons (no rotation)."""
    dx = 1.0
    dy = math.sqrt(3.0) / 2.0
    pts = []
    row = 0
    while len(pts) < N:
        for col in range(-(row + 3), row + 4):
            if len(pts) >= N:
                break
            x = (col + (row % 2) * 0.5) * dx
            y = row * dy
            pts.append((x, y))
        row += 1
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    ang = np.zeros(N)
    return x, y, ang

def _random_layout(S):
    rng = random.Random()
    x = np.array([rng.uniform(-S * 0.45, S * 0.45) for _ in range(N)])
    y = np.array([rng.uniform(-S * 0.45, S * 0.45) for _ in range(N)])
    ang = np.array([rng.uniform(0.0, 60.0) for _ in range(N)])
    return x, y, ang

def construct():
    best_S = 1e9
    best_xyA = None
    s = 1.0

    # start from the honeycomb layout, then random restarts
    starts = [_hex_grid_layout()]
    for _ in range(40):
        starts.append(_random_layout(8.0))

    for x, y, ang in starts:
        if time.time() - T_START > TIME_LIMIT - 10:
            break
        # find a feasible container size for this layout by bisection
        lo, hi = 1.0, 12.0
        if not _feasible(x, y, ang, s, hi):
            continue
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if _feasible(x, y, ang, s, mid):
                hi = mid
            else:
                lo = mid
        S0 = hi
        if S0 >= best_S:
            continue
        S0, bx, by, ba = _optimize(x.copy(), y.copy(), ang.copy(), s, S0, iters=400)
        if S0 < best_S:
            best_S = S0
            best_xyA = (bx, by, ba)

    if best_xyA is None:
        # fallback: guaranteed feasible large container
        x, y, ang = _hex_grid_layout()
        return [[float(x[k]), float(y[k]), float(ang[k])] for k in range(N)], (0.0, 0.0), 40.0, 0.0

    bx, by, ba = best_xyA
    return [[float(bx[k]), float(by[k]), float(ba[k])] for k in range(N)], (0.0, 0.0), float(best_S), 0.0
