"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        th = a + math.radians(60 * k)
        pts.append((cx + scale * math.cos(th), cy + scale * math.sin(th)))
    return np.array(pts)

def _hex_in_hexagon(inner_c, inner_a, outer_c, outer_side, outer_a):
    inner_pts = _hex_vertices(inner_c[0], inner_c[1], inner_a, scale=1.0)
    outer_poly = _hex_vertices(outer_c[0], outer_c[1], outer_a, scale=outer_side)
    n = len(outer_poly)
    for pt in inner_pts:
        for i in range(n):
            a = outer_poly[i]; b = outer_poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross < -1e-9:
                return False
    return True

def _min_side_for_placement(inner, outer_c, outer_a, lo=2.5, hi=8.0, iters=50):
    def feasible(S):
        return all(_hex_in_hexagon(c, a, outer_c, S, outer_a) for (c, a) in inner)
    if not feasible(hi):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi

def _make_block():
    """3x3 hexagonal lattice block: center + 6 ring + 4 corners (11 hexagons).
    Axial coordinates converted to Cartesian. All flat-topped (0 deg)."""
    # axial (q, r) -> cartesian for pointy-top; use flat-topped layout
    # Flat-topped: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    s3 = math.sqrt(3.0)
    cells = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (-1, 2), (0, -2)]
    inner = []
    for (q, r) in cells:
        x = 1.5 * q
        y = s3 * (r + q / 2.0)
        inner.append(((x, y), 0.0))
    return inner

def _optimize_outer(inner):
    best = (None, 8.0)
    for outer_a in [0.0, 30.0, 15.0, 45.0, 60.0]:
        for cx in [-0.4, -0.2, 0.0, 0.2, 0.4]:
            for cy in [-0.4, -0.2, 0.0, 0.2, 0.4]:
                S = _min_side_for_placement(inner, (cx, cy), outer_a, lo=2.5, hi=8.0, iters=40)
                if S < best[1] - 1e-9:
                    best = (((cx, cy), outer_a, S), S)
    (bc, ba, bS) = best[0]
    # local polish on center
    for _ in range(20):
        improved = False
        for dx in [-0.04, -0.02, 0.0, 0.02, 0.04]:
            for dy in [-0.04, -0.02, 0.0, 0.02, 0.04]:
                S = _min_side_for_placement(inner, (bc[0] + dx, bc[1] + dy), ba, lo=2.5, hi=bS + 0.05, iters=30)
                if S < bS - 1e-7:
                    bS = S; bc = (bc[0] + dx, bc[1] + dy); improved = True
        if not improved:
            break
    return bc, ba, bS

def construct():
    t0 = time.time()
    DEADLINE = 140.0

    inner = _make_block()

    if time.time() - t0 > DEADLINE:
        return [[c[0], c[1], a] for (c, a) in inner], (0.0, 0.0), 4.0, 0.0

    bc, ba, bS = _optimize_outer(inner)

    if time.time() - t0 > DEADLINE:
        return [[c[0], c[1], a] for (c, a) in inner], (0.0, 0.0), max(bS, 4.0), 0.0

    return [[c[0], c[1], a] for (c, a) in inner], bc, bS, ba
# EVOLVE-BLOCK-END
