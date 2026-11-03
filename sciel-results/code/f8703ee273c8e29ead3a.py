import math
import time
import itertools

N = 12
DEADLINE = 120.0
START = time.time()


def _elapsed():
    return time.time() - START


def _hex_points(cx, cy, s, a):
    """Vertices of a regular hexagon (flat-top when a=0), side s, rotated a radians."""
    c, sn = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        x = s * math.cos(ang)
        y = s * math.sin(ang)
        pts.append((cx + c * x - sn * y, cy + sn * x + c * y))
    return pts


def _point_in_hex(px, py, cx, cy, s, a):
    """True if (px,py) is inside the regular hexagon (flat-top, side s, rotated a)."""
    c, sn = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    u = c * dx + sn * dy
    v = -sn * dx + c * dy
    s3 = 3.0 * s
    return (abs(u) <= s and abs(v) <= s3 and abs(2.0 * v + u) <= s3
            and abs(2.0 * v - u) <= s3)


def _min_enclosing_hex(sides, center, a):
    """Side length of the minimal regular hexagon (flat-top, rotated a) with given
    center that contains every polygon in `sides` (list of point lists)."""
    c, sn = math.cos(a), math.sin(a)
    cu, sv, cuu, svv = 0.0, 0.0, 0.0, 0.0
    for poly in sides:
        for (x, y) in poly:
            u = c * (x - center[0]) + sn * (y - center[1])
            v = -sn * (x - center[0]) + c * (y - center[1])
            au, av = abs(u), abs(v)
            cu += au
            sv += av
            cuu += max(au, 2.0 * av)
            svv += max(av, 2.0 * au)
    return max(cu, sv / 3.0, cuu / 3.0, svv / 3.0)


def _min_enclosing_hex_search(sides, center, n=1500):
    """1-D golden-section search over outer rotation angle in [0, 30) degrees."""
    lo, hi = 0.0, math.radians(30.0)
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c1 = hi - phi * (hi - lo)
    c2 = lo + phi * (hi - lo)
    f1 = _min_enclosing_hex(sides, center, c1)
    f2 = _min_enclosing_hex(sides, center, c2)
    for _ in range(n):
        if f1 < f2:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - phi * (hi - lo)
            f1 = _min_enclosing_hex(sides, center, c1)
        else:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + phi * (hi - lo)
            f2 = _min_enclosing_hex(sides, center, c2)
    return min(f1, f2)


def _build_cluster(ring_counts):
    """Place N unit hexagons on the hex lattice (flat-top, side 1). ring_counts is a
    list of how many to take from ring 0, ring 1, ring 2, ... Returns (points, centers)."""
    centers = []
    for ring, count in enumerate(ring_counts):
        if count <= 0:
            continue
        if ring == 0:
            ring_pts = [(0, 0)]
        else:
            ring_pts = []
            for k in range(6 * ring):
                a = math.radians(60.0 * k + 30.0)
                ring_pts.append((ring * math.cos(a), ring * math.sin(a)))
        centers.extend(ring_pts[:count])
        if len(centers) >= N:
            break
    sides = [_hex_points(x, y, 1.0, 0.0) for (x, y) in centers]
    return sides, centers


def _make_result(centers, center, side):
    inner = [[float(x), float(y), 0.0] for (x, y) in centers]
    return inner, (float(center[0]), float(center[1])), float(side), 0.0


def construct():
    t0 = time.time()

    # Candidate cluster configurations (ring 0, ring 1, ring 2, ...)
    configs = [
        (1, 6, 5),   # flower (13 cells) minus one outer
        (1, 6, 6),   # full flower (13 cells) minus center
        (1, 5, 6),   # center + 11
        (0, 6, 6),   # 12 outer, no center
        (1, 6, 4),   # center + 6 + 4
        (0, 5, 7),
    ]

    best = None
    for cfg in configs:
        if time.time() - t0 > DEADLINE - 10.0:
            break
        sides, centers = _build_cluster(cfg)
        if len(centers) != N:
            continue
        # center = centroid of the placed hexagon centers (near-optimal for enclosing)
        mx = sum(x for (x, y) in centers) / N
        my = sum(y for (x, y) in centers) / N
        center = (mx, my)
        side = _min_enclosing_hex_search(sides, center, n=1200)
        if best is None or side < best[0]:
            best = (side, centers, center)
        if time.time() - t0 > DEADLINE - 10.0:
            break

    if best is None:
        # Fallback: trivially valid collinear placement (seed-like) with a safe side.
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    side, centers, center = best
    return _make_result(centers, center, side)
