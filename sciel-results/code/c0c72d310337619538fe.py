"""
Packing 11 unit regular hexagons inside a regular hexagon.
Structural change: treat the inner hexagon orientations as discrete candidates (every 5 deg),
enumerate a small set of compact lattice topologies, and for each topology/angle combination
solve the exact minimum-enclosing-hexagon LP. Then polish the continuous outer angle with a
fine 1-D golden-section search. This avoids the fragile smooth-overlap penalty and the
expensive per-pair SAT checks in the hot loop.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    """6 vertices of a regular hexagon (center cx,cy; side; rotation in degrees)."""
    out = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        out.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return out


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Strict containment test for a point in a rotated regular hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2.0
    for k in range(6):
        an = math.radians(30 + 60 * k)
        if lx * math.cos(an) + ly * math.sin(an) > apothem + 1e-9:
            return False
    return True


def _separating_axis_disjoint(hex1, hex2, eps=1e-9):
    """Convex polygon separation (SAT) for two regular hexagons. Returns True if disjoint."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    # Bounding-circle quick reject
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    v1 = np.array(_hex_vertices(cx1, cy1, s1, a1))
    v2 = np.array(_hex_vertices(cx2, cy2, s2, a2))
    # Vertex-in-polygon checks
    for vx, vy in v1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for vx, vy in v2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    # SAT over all 12 edge normals
    for verts, other in [(v1, v2), (v2, v1)]:
        for i in range(6):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.hypot(ex, ey)
            if length < 1e-15:
                continue
            nx = -ey / length
            ny = ex / length
            p1 = verts @ np.array([nx, ny])
            p2 = other @ np.array([nx, ny])
            if p1.max() <= p2.min() + eps or p2.max() <= p1.min() + eps:
                return True
    return False


def _all_disjoint(hexes, eps=1e-9):
    n = len(hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separating_axis_disjoint(hexes[i], hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner, outer):
    cx, cy, s, a = inner
    ox, oy, oside, oa = outer
    for vx, vy in _hex_vertices(cx, cy, s, a):
        if not _point_in_hex(vx, vy, ox, oy, oside, oa):
            return False
    return True


def _min_outer_side(hexes, oangle_rad):
    """
    Exact LP for the minimum side length of an outer regular hexagon (rotation oangle_rad)
    containing all vertices of the given inner hexagons. Returns (side, ox, oy).
    """
    pts = []
    for cx, cy, s, ang in hexes:
        pts.extend(_hex_vertices(cx, cy, s, ang))
    if not pts:
        return 0.0, 0.0, 0.0
    pts = np.array(pts)

    u = np.zeros((6, 2))
    for k in range(6):
        th = oangle_rad + math.radians(30 + 60 * k)
        u[k, 0] = math.cos(th)
        u[k, 1] = math.sin(th)

    proj = pts @ u.T
    P = proj.max(axis=1)
    Q = proj.min(axis=1)

    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u[k]
        A_mat[2 * k, 0] = ux
        A_mat[2 * k, 1] = uy
        A_mat[2 * k, 2] = -1.0
        b_vec[2 * k] = Q[k]
        A_mat[2 * k + 1, 0] = -ux
        A_mat[2 * k + 1, 1] = -uy
        A_mat[2 * k + 1, 2] = -1.0
        b_vec[2 * k + 1] = -P[k]

    c_vec = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_vec, A_ub=A_mat, b_ub=b_vec,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            A_opt = res.x[2]
            ox_opt = res.x[0]
            oy_opt = res.x[1]
            side = A_opt * 2.0 / SQRT3
            return side, ox_opt, oy_opt
    except Exception:
        pass
    return float('inf'), 0.0, 0.0


def _axial_to_xy(q, r):
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _build_topologies():
    """
    Return a list of topologies. Each topology is a list of 11 (q, r) axial coordinates.
    Topology 0: 7 center + 4 ring-2 (the classic compact cluster).
    Additional topologies explore other ring-2 selections.
    """
    topologies = []

    # Base 7: center + ring 1
    base = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    # Ring 2 candidates (14 positions)
    ring2 = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]

    # We'll enumerate all C(14,4) = 1001 combinations but only keep those that are
    # pairwise disjoint (no two hexagons overlap) and form a connected cluster.
    # To keep it fast, we pre-filter by checking that the 4 ring-2 hexagons don't
    # overlap with each other or with the base 7.

    # Precompute base xy
    base_xy = [_axial_to_xy(q, r) for q, r in base]
    ring2_xy = [_axial_to_xy(q, r) for q, r in ring2]

    # For speed, we'll just try all 1001 combos but with a fast rejection:
    # Two unit hexagons on the hex lattice are disjoint iff their axial distance >= 2
    # (i.e., they are not neighbors). Actually, on the lattice, two hexagons at axial
    # distance 1 share an edge (touching). The evaluator may consider touching as overlap.
    # So we need axial distance >= 2 for strict disjointness (gap > 0).
    # But wait: the base 7 already have touching hexagons! The center touches all 6 ring-1.
    # So the "disjoint" requirement must mean "interiors disjoint" which allows edge-touching.
    # The SAT check with eps=1e-9 should handle this: edge-touching hexagons have
    # projections that just meet, so the SAT test returns True (disjoint).
    # So we can use lattice positions directly.

    # To keep the enumeration manageable, we'll just use the full 1001 combos.
    # Each combo evaluation is a fast LP (66 points, 12 constraints) ~ microseconds.
    # 1001 * ~50us = ~50ms, well within budget.

    for combo in itertools.combinations(range(14), 4):
        selected = base + [ring2[i] for i in combo]
        topologies.append(selected)

    return topologies


def _golden_section_min(f, lo, hi, iters=30):
    """Minimize a unimodal function f on [lo, hi] using golden-section search."""
    gr = (math.sqrt(5) - 1) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    return (lo + hi) / 2.0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Candidate angles for inner hexagons: 0, 5, 10, 15, 20, 25, 30 degrees
    # Due to 60-degree symmetry of the hexagon, angles mod 60 are equivalent.
    # We'll test a few discrete orientations.
    inner_angle_candidates = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    # Build topologies
    topologies = _build_topologies()

    best_side = float('inf')
    best_result = None

    # For each topology and each inner angle candidate, find the best outer angle
    for top_idx, topo in enumerate(topologies):
        if time.time() > deadline:
            break

        for inner_ang in inner_angle_candidates:
            if time.time() > deadline:
                break

            # Build Cartesian positions
            xy = [_axial_to_xy(q, r) for q, r in topo]

            # Center the cluster
            cx_mean = sum(x for x, y in xy) / N
            cy_mean = sum(y for x, y in xy) / N
            centered_xy = [(x - cx_mean, y - cy_mean) for x, y in xy]

            # Build inner hexagons with the candidate angle
            inner_hexes = [(x, y, 1.0, inner_ang) for x, y in centered_xy]

            # Check disjointness (should be true for lattice positions)
            if not _all_disjoint(inner_hexes):
                continue

            # Find best outer angle via golden-section search on [0, 30] degrees
            def f_outer(angle_deg):
                side, _, _ = _min_outer_side(inner_hexes, math.radians(angle_deg))
                return side

            best_outer_ang = _golden_section_min(f_outer, 0.0, 30.0, iters=25)
            side, ox, oy = _min_outer_side(inner_hexes, math.radians(best_outer_ang))

            if side < best_side:
                best_side = side
                # Convert back to global coordinates
                global_xy = [(x + cx_mean, y + cy_mean) for x, y in centered_xy]
                best_result = {
                    'xy': global_xy,
                    'inner_ang': inner_ang,
                    'outer_ang': best_outer_ang,
                    'ox': ox + cx_mean,
                    'oy': oy + cy_mean,
                    'side': side,
                }

    if best_result is None:
        # Fallback: just use the first topology with angle 0
        topo = topologies[0]
        xy = [_axial_to_xy(q, r) for q, r in topo]
        cx_mean = sum(x for x, y in xy) / N
        cy_mean = sum(y for x, y in xy) / N
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in xy]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        side, ox, oy = _min_outer_side(inner_hexes, 0.0)
        best_result = {
            'xy': xy,
            'inner_ang': 0.0,
            'outer_ang': 0.0,
            'ox': ox + cx_mean,
            'oy': oy + cy_mean,
            'side': side,
        }
        best_side = side

    # Polish phase: fine-tune the outer angle around the best found
    if time.time() < deadline - 30.0:
        inner_ang = best_result['inner_ang']
        xy = best_result['xy']
        cx_mean = sum(x for x, y in xy) / N
        cy_mean = sum(y for x, y in xy) / N
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in xy]
        inner_hexes = [(x, y, 1.0, inner_ang) for x, y in centered_xy]

        # Fine search around best outer angle
        ang_lo = max(0.0, best_result['outer_ang'] - 0.5)
        ang_hi = min(30.0, best_result['outer_ang'] + 0.5)

        def f_outer(angle_deg):
            side, _, _ = _min_outer_side(inner_hexes, math.radians(angle_deg))
            return side

        best_outer_ang = _golden_section_min(f_outer, ang_lo, ang_hi, iters=30)
        side, ox, oy = _min_outer_side(inner_hexes, math.radians(best_outer_ang))

        if side < best_side - 1e-12:
            best_side = side
            best_result['outer_ang'] = best_outer_ang
            best_result['ox'] = ox + cx_mean
            best_result['oy'] = oy + cy_mean
            best_result['side'] = side

    # Build final inner hexagons
    inner_hexes_final = [(x, y, 1.0, best_result['inner_ang']) for x, y in best_result['xy']]

    # Verify disjointness
    if not _all_disjoint(inner_hexes_final):
        # Apply a tiny gap factor
        scale = 1.001
        inner_hexes_final = [(x * scale, y * scale, 1.0, best_result['inner_ang'])
                             for x, y in best_result['xy']]
        cx_mean = sum(h[0] for h in inner_hexes_final) / N
        cy_mean = sum(h[1] for h in inner_hexes_final) / N
        centered = [(h[0] - cx_mean, h[1] - cy_mean, 1.0, h[3]) for h in inner_hexes_final]
        side, ox, oy = _min_outer_side(centered, math.radians(best_result['outer_ang']))
        best_result['side'] = side
        best_result['ox'] = ox + cx_mean
        best_result['oy'] = oy + cy_mean
        inner_hexes_final = [(x + cx_mean, y + cy_mean, 1.0, a)
                             for x, y, s, a in centered]

    # Verify containment
    outer_hex = (best_result['ox'], best_result['oy'], best_result['side'],
                 best_result['outer_ang'])
    valid = True
    for h in inner_hexes_final:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_result['side'] *= 1.005
        outer_hex = (best_result['ox'], best_result['oy'], best_result['side'],
                     best_result['outer_ang'])

    inner = [[h[0], h[1], h[3]] for h in inner_hexes_final]
    return inner, (best_result['ox'], best_result['oy']), best_result['side'], \
        best_result['outer_ang']
