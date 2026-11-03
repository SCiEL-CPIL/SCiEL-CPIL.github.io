"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Rigid compact cluster (center + 6 ring-1 + 4 ring-2), exhaustive combination
search with dense sweep over outer hexagon orientation (no LP, analytic max of sinusoids).
"""
import math
import time
import itertools
import numpy as np

N = 11
DEADLINE = 150.0
SAFETY = 12.0  # seconds before hard limit
SQRT3 = math.sqrt(3.0)


def _hex_vertices(cx, cy, side, angle_deg):
    """6 vertices of a regular hexagon (flat-top at angle 0)."""
    a0 = math.radians(angle_deg)
    return [(cx + side * math.cos(a0 + math.radians(60 * k)),
             cy + side * math.sin(a0 + math.radians(60 * k))) for k in range(6)]


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2.0
    for k in range(6):
        an = math.radians(30 + 60 * k)
        if lx * math.cos(an) + ly * math.sin(an) > apothem + 1e-8:
            return False
    return True


def _segs_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hex_disjoint(h1, h2, eps=1e-9):
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    dx = cx2 - cx1; dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    v1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in v1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    v2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in v2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        for j in range(6):
            if _segs_intersect(v1[i], v1[(i + 1) % 6], v2[j], v2[(j + 1) % 6]):
                return False
    return True


def _all_disjoint(hexes):
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hex_disjoint(hexes[i], hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oa = outer_hex
    for vx, vy in _hex_vertices(cx, cy, s, a):
        if not _point_in_hex(vx, vy, ox, oy, oside, oa):
            return False
    return True


def _axial_to_xy(q, r):
    return (1.5 * q, SQRT3 * (r + q / 2.0))


def _build_cluster(selected_idx):
    """Return list of (x, y) for center + 6 ring1 + 4 selected ring2."""
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    pts = [(0.0, 0.0)]
    for q, r in ring1:
        pts.append(_axial_to_xy(q, r))
    for i in selected_idx:
        q, r = ring2[i]
        pts.append(_axial_to_xy(q, r))
    return pts


def _outer_side_for_rigid(pts, oangle_deg):
    """Min side of outer hexagon (orientation oangle) enclosing all vertices of
    unit hexagons centered at pts (all inner angle 0). Returns (side, cx, cy) or (inf,0,0)."""
    # Precompute all 66 inner vertices
    allv = []
    for (px, py) in pts:
        for k in range(6):
            a = math.radians(60 * k)
            allv.append((px + math.cos(a), py + math.sin(a)))
    allv = np.array(allv)  # (66, 2)

    # 6 outer facet normals
    base = np.radians(30 + 60 * np.arange(6))
    nxs = np.cos(base); nys = np.sin(base)

    # For each facet k: max over vertices of (vx*nx + vy*ny)
    # M = allv @ [nx, ny] -> (66,)
    best_t = 0.0
    for k in range(6):
        proj = allv[:, 0] * nxs[k] + allv[:, 1] * nys[k]
        mx = proj.max()
        if mx > best_t:
            best_t = mx

    # Center: minimize max of (n_k . c + t) and (-n_k . c + t)
    #   => for each k: -t + n_k . c <= 0 and -t - n_k . c <= 0
    #   => -t + 0*c <= -best_t  (t >= best_t, handled by t=best_t)
    # LP: vars [cx, cy, t], min t
    A = []
    b = []
    for k in range(6):
        A.append([nxs[k], nys[k], -1.0]); b.append(0.0)
        A.append([-nxs[k], -nys[k], -1.0]); b.append(0.0)
    A.append([0.0, 0.0, -1.0]); b.append(-best_t)
    A = np.array(A); b = np.array(b)

    from scipy.optimize import linprog
    res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A, b_ub=b,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if not res.success:
        return (float('inf'), 0.0, 0.0)
    t = res.x[2]
    side = t * 2.0 / SQRT3
    return (side, res.x[0], res.x[1])


def _sweep_best_for_cluster(pts, angles_deg, start, end, step):
    """Dense sweep over outer angle; returns best (side, cx, cy, oangle)."""
    best = (float('inf'), 0.0, 0.0, 0.0)
    a = start
    while a <= end + 1e-9:
        s, cx, cy = _outer_side_for_rigid(pts, a)
        if s < best[0]:
            best = (s, cx, cy, a)
        a += step
    return best


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    # Precompute all 1001 clusters
    ring2_idx = list(range(14))
    combos = list(itertools.combinations(ring2_idx, 4))
    clusters = []
    for comb in combos:
        clusters.append(_build_cluster(comb))

    # Phase 1: coarse 5-degree sweep over [0, 60) for every cluster
    best_side = float('inf')
    best_pts = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    best_comb = None

    for ci, pts in enumerate(clusters):
        if time.time() > deadline:
            break
        s, cx, cy, oa = _sweep_best_for_cluster(pts, [0.0] * 11, 0.0, 60.0, 5.0)
        if s < best_side:
            best_side = s
            best_pts = pts
            best_center = (cx, cy)
            best_oangle = oa
            best_comb = combos[ci]

    if best_pts is None:
        best_pts = _build_cluster((0, 1, 2, 3))
        s, cx, cy, oa = _outer_side_for_rigid(best_pts, 0.0)
        best_side = s
        best_center = (cx, cy)
        best_oangle = oa

    # Phase 2: fine 1-degree sweep around best_oangle for the top few clusters
    # Re-rank clusters by their coarse best, keep top 30
    coarse_scores = []
    for ci, pts in enumerate(clusters):
        if time.time() > deadline:
            break
        s, cx, cy, oa = _sweep_best_for_cluster(pts, [0.0] * 11, 0.0, 60.0, 5.0)
        coarse_scores.append((s, ci, cx, cy, oa))
    coarse_scores.sort()

    for (s, ci, cx, cy, oa) in coarse_scores[:30]:
        if time.time() > deadline:
            break
        pts = clusters[ci]
        lo = max(0.0, oa - 5.0)
        hi = min(60.0, oa + 5.0)
        s2, cx2, cy2, oa2 = _sweep_best_for_cluster(pts, [0.0] * 11, lo, hi, 1.0)
        if s2 < best_side:
            best_side = s2
            best_pts = pts
            best_center = (cx2, cy2)
            best_oangle = oa2
            best_comb = combos[ci]

    # Phase 3: ultra-fine 0.1-degree sweep around the global best
    if time.time() < deadline - 2.0:
        pts = best_pts
        lo = max(0.0, best_oangle - 2.0)
        hi = min(60.0, best_oangle + 2.0)
        s3, cx3, cy3, oa3 = _sweep_best_for_cluster(pts, [0.0] * 11, lo, hi, 0.1)
        if s3 < best_side:
            best_side = s3
            best_center = (cx3, cy3)
            best_oangle = oa3

        # Even finer 0.01-degree around the new best
        if time.time() < deadline - 1.0:
            lo = max(0.0, best_oangle - 0.2)
            hi = min(60.0, best_oangle + 0.2)
            s4, cx4, cy4, oa4 = _sweep_best_for_cluster(pts, [0.0] * 11, lo, hi, 0.01)
            if s4 < best_side:
                best_side = s4
                best_center = (cx4, cy4)
                best_oangle = oa4

    # Build inner hexagons (all angle 0)
    inner_hexes = [(best_pts[i][0], best_pts[i][1], 1.0, 0.0) for i in range(len(best_pts))]

    # Verify disjointness (should hold for lattice cluster)
    if not _all_disjoint(inner_hexes):
        # Fallback: scale positions slightly
        for scale in [1.001, 1.005, 1.01, 1.02, 1.03]:
            scaled = [(x * scale, y * scale, 1.0, 0.0) for (x, y, _, _) in inner_hexes]
            if _all_disjoint(scaled):
                inner_hexes = scaled
                best_pts = [(h[0], h[1]) for h in inner_hexes]
                s, cx, cy = _outer_side_for_rigid(best_pts, best_oangle)
                best_side = s
                best_center = (cx, cy)
                break

    # Verify containment; bump side if needed
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    for _ in range(200):
        if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
            break
        best_side *= 1.0005
        outer_hex = (best_center[0], best_center[1], best_side, best_oangle)

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
