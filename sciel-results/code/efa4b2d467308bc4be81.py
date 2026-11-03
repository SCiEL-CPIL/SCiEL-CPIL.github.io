"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses rotated lattice structures (15° and 30° rotations) with multi-stage optimization.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 15.0
EPS = 1e-9
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    return [(cx + side * math.cos(math.radians(angle_deg + 60 * k)),
             cy + side * math.sin(math.radians(angle_deg + 60 * k)))
            for k in range(6)]


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * SQRT3 / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        if lx * math.cos(a_norm) + ly * math.sin(a_norm) > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segments properly intersect."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
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
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side_lp(inner_hexes, oangle=0.0):
    """Find minimum side length of outer hexagon using LP."""
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
        return t * 2.0 / SQRT3
    return float('inf')


def _compute_outer_for_positions(positions, angles, oangle=0.0):
    """Compute optimal outer hexagon for given inner positions and angles."""
    inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i])
                   for i in range(len(positions))]
    side = _min_outer_side_lp(inner_hexes, oangle)
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
        center = (res.x[0], res.x[1])
    else:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    return center, side


def _support_function(cx, cy, s, ang, nx, ny):
    """Support function of hexagon in direction (nx, ny)."""
    max_val = 0.0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _pairwise_violation(pos, angles, i, j):
    """Return positive violation if hexagons i, j overlap; 0 if disjoint."""
    max_violation = 0.0
    for k in range(12):
        theta = math.radians(15 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        s_i = _support_function(pos[2*i], pos[2*i+1], 1.0, angles[i], nx, ny)
        s_j = _support_function(pos[2*j], pos[2*j+1], 1.0, angles[j], -nx, -ny)
        dx = pos[2*j] - pos[2*i]
        dy = pos[2*j+1] - pos[2*i+1]
        dist_proj = dx * nx + dy * ny
        violation = s_i + s_j - dist_proj
        if violation > max_violation:
            max_violation = violation
    return max_violation


def _build_rotated_lattice(rotation_deg):
    """
    Build 11 hexagon positions on a hexagonal lattice, rotated by rotation_deg.
    Returns list of (x, y) positions.
    """
    # Standard hexagonal lattice positions for 11 hexagons
    # Center + 6 in ring 1 + 4 in ring 2
    axial = [(0, 0),
             (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
             (2, 0), (2, -1), (1, -2), (0, -2)]
    
    positions = []
    for q, r in axial:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        # Rotate around origin
        theta = math.radians(rotation_deg)
        xr = x * math.cos(theta) - y * math.sin(theta)
        yr = x * math.sin(theta) + y * math.cos(theta)
        positions.append((xr, yr))
    return positions


def _objective_function(state, n_hex, oangle, penalty_weight=200.0):
    """
    Objective: outer side length + penalty for overlaps.
    state: flat [x1,y1,a1, x2,y2,a2, ...]
    """
    inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                   for i in range(n_hex)]
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
    if not res.success:
        return 1e6
    t = res.x[2]
    side = t * 2.0 / SQRT3
    
    angles_list = [state[3*m+2] for m in range(n_hex)]
    penalty = 0.0
    for i in range(n_hex):
        for j in range(i + 1, n_hex):
            v = _pairwise_violation(state, angles_list, i, j)
            if v > 0:
                penalty += v ** 2
    return side + penalty_weight * penalty


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Try multiple rotated lattice configurations
    # This is the key structural change: explore 15° and 30° rotations
    lattice_rotations = [0.0, 15.0, 30.0, -15.0, -30.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0]
    oangle_options = [0.0, 15.0, 30.0, -15.0, -30.0]

    for rot in lattice_rotations:
        if time.time() > deadline:
            break
        positions = _build_rotated_lattice(rot)
        angles = [rot] * 11  # All inner hexagons rotated same as lattice
        
        for oangle in oangle_options:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(positions, angles, oangle)
            if side < best_side:
                best_side = side
                best_positions = positions[:]
                best_angles = angles[:]
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        positions = _build_rotated_lattice(0.0)
        best_positions = positions
        best_angles = [0.0] * 11
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    # Phase 2: Nonlinear optimization from best lattice configuration
    if time.time() < deadline - 10:
        n_hex = len(best_positions)
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(_objective_function, init_state,
                             args=(n_hex, best_oangle, 200.0),
                             method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_side + 100:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Phase 3: Try different outer angles with current best positions
    if time.time() < deadline - 15:
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
                       -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 4: High-precision polish
    if time.time() < deadline - 5:
        n_hex = len(best_positions)
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(_objective_function, init_state,
                             args=(n_hex, best_oangle, 500.0),
                             method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 20:
                final_state = result.x
                hexes = [(final_state[3*i], final_state[3*i+1], 1.0, final_state[3*i+2])
                         for i in range(n_hex)]
                if _all_disjoint(hexes):
                    positions = [(h[0], h[1]) for h in hexes]
                    angles = [h[3] for h in hexes]
                    center, side = _compute_outer_for_positions(positions, angles, best_oangle)
                    if side < best_side:
                        best_side = side
                        best_positions = positions
                        best_angles = angles
                        best_center = center
        except Exception:
            pass

    # Final verification and safety scaling
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i])
                   for i in range(len(best_positions))]

    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(
                    best_positions, best_angles, best_oangle)
                break

    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        for _ in range(200):
            best_side *= 1.001
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break

    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]]
             for i in range(len(inner_hexes))]

    return inner, (best_center[0], best_center[1]), best_side, best_oangle
