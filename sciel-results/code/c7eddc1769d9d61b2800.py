"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses a 1D angular sweep over the rigid honeycomb cluster, with a final
high-precision polish pass on the outer orientation.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _outer_lp(inner_hexes, oangle):
    """
    LP to find minimum enclosing regular hexagon (center + side) for fixed oangle.
    Returns (ox, oy, side) or (None, None, inf).
    """
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    for cx_h, cy_h, s_h, ang_h in inner_hexes:
        verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)], method='highs')
    if res.success:
        t = res.x[2]
        return (res.x[0], res.x[1], t * 2.0 / SQRT3)
    return (None, None, float('inf'))


def _build_honeycomb():
    """Standard 11-hex honeycomb: center + ring1(6) + ring2(4)."""
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
             (2, 0), (0, -2), (-2, -1), (-1, 2)]
    xy = [axial_to_xy(q, r) for q, r in axial]
    return [(x, y) for x, y in xy]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    best_state = None

    # Build the rigid honeycomb cluster
    hc = _build_honeycomb()
    n_hex = 11

    # Precompute the cluster's vertex positions in local coordinates
    # All inner hexagons have angle 0 in the cluster frame
    local_positions = [(x, y) for x, y in hc]

    # For each cluster rotation theta, the global positions are rotated by theta
    # and the outer hexagon has a fixed orientation (0 degrees).
    # We sweep theta to minimize the outer side length.

    def cluster_side(theta_deg):
        """Compute the minimum outer side for the cluster rotated by theta_deg."""
        theta = math.radians(theta_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        inner_hexes = []
        for x, y in local_positions:
            gx = x * cos_t - y * sin_t
            gy = x * sin_t + y * cos_t
            inner_hexes.append((gx, gy, 1.0, theta_deg))
        ox, oy, side = _outer_lp(inner_hexes, 0.0)
        if ox is None:
            return float('inf')
        return side

    # Phase 1: Coarse angular sweep
    if time_left() > 30:
        theta_coarse = np.arange(-30, 30.5, 0.5)
        for t in theta_coarse:
            if time_left() < 30:
                break
            s = cluster_side(float(t))
            if s < best_side:
                best_side = s
                best_oangle = 0.0
                # Reconstruct best state
                theta = math.radians(float(t))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                best_state = []
                for x, y in local_positions:
                    gx = x * cos_t - y * sin_t
                    gy = x * sin_t + y * cos_t
                    best_state.extend([gx, gy, float(t)])
                ox, oy, _ = _outer_lp(
                    [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)],
                    0.0)
                best_center = (ox, oy)

    # Phase 2: Fine angular sweep around best
    if time_left() > 20 and best_state is not None:
        # Find the best coarse angle
        theta_best = best_state[0]  # This is the x of first hex, not the angle
        # Actually, let's re-extract the angle from best_state
        theta_best = best_state[2]  # angle of first hex
        theta_fine = np.arange(theta_best - 1.0, theta_best + 1.05, 0.05)
        for t in theta_fine:
            if time_left() < 20:
                break
            s = cluster_side(float(t))
            if s < best_side:
                best_side = s
                best_oangle = 0.0
                theta = math.radians(float(t))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                best_state = []
                for x, y in local_positions:
                    gx = x * cos_t - y * sin_t
                    gy = x * sin_t + y * cos_t
                    best_state.extend([gx, gy, float(t)])
                ox, oy, _ = _outer_lp(
                    [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)],
                    0.0)
                best_center = (ox, oy)

    # Phase 3: Very fine angular sweep
    if time_left() > 15 and best_state is not None:
        theta_best = best_state[2]
        theta_vfine = np.arange(theta_best - 0.1, theta_best + 0.105, 0.005)
        for t in theta_vfine:
            if time_left() < 15:
                break
            s = cluster_side(float(t))
            if s < best_side:
                best_side = s
                best_oangle = 0.0
                theta = math.radians(float(t))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                best_state = []
                for x, y in local_positions:
                    gx = x * cos_t - y * sin_t
                    gy = x * sin_t + y * cos_t
                    best_state.extend([gx, gy, float(t)])
                ox, oy, _ = _outer_lp(
                    [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)],
                    0.0)
                best_center = (ox, oy)

    # Phase 4: Use minimize_scalar for ultra-fine polish
    if time_left() > 10 and best_state is not None:
        theta_best = best_state[2]

        def obj(theta_deg):
            return cluster_side(float(theta_deg))

        try:
            res = minimize_scalar(obj, bounds=(theta_best - 0.01, theta_best + 0.01),
                                  method='bounded', options={'xatol': 1e-10})
            if res.fun < best_side:
                best_side = res.fun
                best_oangle = 0.0
                theta = math.radians(float(res.x))
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                best_state = []
                for x, y in local_positions:
                    gx = x * cos_t - y * sin_t
                    gy = x * sin_t + y * cos_t
                    best_state.extend([gx, gy, float(res.x)])
                ox, oy, _ = _outer_lp(
                    [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)],
                    0.0)
                best_center = (ox, oy)
        except Exception:
            pass

    # Final verification and fallback
    if best_state is None:
        # Fallback to honeycomb at angle 0
        best_state = []
        for x, y in local_positions:
            best_state.extend([x, y, 0.0])
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in local_positions]
        ox, oy, side = _outer_lp(inner_hexes, 0.0)
        best_center = (ox, oy)
        best_side = side
        best_oangle = 0.0

    inner_hexes = [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)]
    if not _all_disjoint(inner_hexes):
        # This shouldn't happen for the rigid honeycomb, but just in case
        for scale in [1.001, 1.002, 1.005, 1.01, 1.02]:
            inner_hexes = [(best_state[3*i]*scale, best_state[3*i+1]*scale, 1.0, best_state[3*i+2]) for i in range(n_hex)]
            if _all_disjoint(inner_hexes):
                best_state = []
                for x, y, s, a in inner_hexes:
                    best_state.extend([x, y, a])
                break

    inner_hexes = [(best_state[3*i], best_state[3*i+1], 1.0, best_state[3*i+2]) for i in range(n_hex)]
    ox, oy, side = _outer_lp(inner_hexes, best_oangle)
    if ox is not None:
        best_center = (ox, oy)
        best_side = side

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = all(_inner_in_outer(h, outer_hex) for h in inner_hexes)
    if not valid:
        for _ in range(500):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[best_state[3*i], best_state[3*i+1], best_state[3*i+2]] for i in range(n_hex)]
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
