"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Graph-embedding (spring-mass) initialization with contact-graph topology,
followed by smooth-surrogate optimization and LP-based outer hexagon refinement.
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
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _hex_vertices_array(cx, cy, side, angle_deg):
    """Return 6 vertices as numpy array."""
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


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
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
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


def _pairwise_violation(pos, angles, i, j, n_dirs=12):
    """Return positive violation if hexagons i, j overlap; 0 if disjoint."""
    max_violation = 0.0
    for k in range(n_dirs):
        theta = 2.0 * math.pi * k / n_dirs
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


def _build_honeycomb_layout():
    """
    Build the standard honeycomb cluster of 11 hexagons:
    center + ring1 (6) + ring2 (4 selected for compactness).
    Returns list of (x, y, angle) for 11 unit hexagons.
    """
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    # Center
    pos = [(0, 0)]
    # Ring 1: 6 hexagons
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos.extend(ring1)
    # Ring 2: select 4 for compact 11-hex cluster
    # Use positions that form a roughly circular cluster
    ring2_candidates = [(2, 0), (2, -1), (1, -2), (0, -2), (-1, -2),
                        (-2, -1), (-2, 0), (-1, 1), (0, 1), (1, 1), (2, -1), (2, 0)]
    # Pick 4 that are well-distributed
    ring2_sel = [(2, 0), (0, -2), (-2, -1), (-1, 2)]
    pos.extend(ring2_sel)

    xy = [axial_to_xy(q, r) for q, r in pos]
    return [(x, y, 0.0) for x, y in xy]


def _spring_mass_relax(positions, angles, contact_graph, n_iter=500, lr=0.01):
    """
    Relax positions using spring-mass with contact graph.
    Rest length for edge-to-edge touching hexagons = 2*apothem = sqrt(3).
    Also add repulsion for non-contact pairs to prevent overlap.
    """
    n = len(positions)
    rest_length = SQRT3  # distance between centers of touching hexagons
    pos = np.array([[positions[i][0], positions[i][1]] for i in range(n)], dtype=float)
    ang = np.array(angles, dtype=float)

    for it in range(n_iter):
        forces = np.zeros_like(pos)
        # Spring forces along contact graph
        for i, j in contact_graph:
            dx = pos[j][0] - pos[i][0]
            dy = pos[j][1] - pos[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                continue
            fx = lr * (dist - rest_length) * dx / dist
            fy = lr * (dist - rest_length) * dy / dist
            forces[i][0] -= fx
            forces[i][1] -= fy
            forces[j][0] += fx
            forces[j][1] += fy

        # Repulsion for all pairs (stronger when close)
        for i in range(n):
            for j in range(i+1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    continue
                if dist < rest_length * 1.1:
                    # Strong repulsion
                    f = 0.5 * (rest_length - dist)
                    forces[i][0] -= f * dx / dist
                    forces[i][1] -= f * dy / dist
                    forces[j][0] += f * dx / dist
                    forces[j][1] += f * dy / dist

        pos += forces
        # Damping
        pos *= 0.999

    return [(pos[i][0], pos[i][1], ang[i]) for i in range(n)]


def _get_honeycomb_contact_graph():
    """
    Define contact graph for 11-hex honeycomb cluster.
    Node 0 = center, 1-6 = ring1, 7-10 = ring2.
    """
    graph = []
    # Center connected to all ring1
    for i in range(1, 7):
        graph.append((0, i))
    # Ring1 adjacent pairs
    for i in range(1, 7):
        j = i + 1 if i < 6 else 1
        graph.append((i, j))
    # Ring2 connections to ring1 (approximate)
    # ring2[0]=(2,0) connects to ring1[0]=(1,0) and ring1[1]=(1,-1)
    graph.append((7, 1))
    graph.append((7, 2))
    # ring2[1]=(0,-2) connects to ring1[2]=(0,-1) and ring1[3]=(-1,0)
    graph.append((8, 3))
    graph.append((8, 4))
    # ring2[2]=(-2,-1) connects to ring1[4]=(-1,0) and ring1[5]=(-1,1)
    graph.append((9, 4))
    graph.append((9, 5))
    # ring2[3]=(-1,2) connects to ring1[5]=(-1,1) and ring1[6]=(0,1)
    graph.append((10, 5))
    graph.append((10, 6))
    return graph


def _smooth_outer_side(state, n_hex, oangle, temperature=0.01):
    """
    Smooth log-sum-exp surrogate for the outer hexagon side length.
    state: flat [x1,y1,a1, x2,y2,a2, ...]
    """
    vals = []
    for i in range(n_hex):
        cx = state[3*i]
        cy = state[3*i+1]
        ang = state[3*i+2]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            for k in range(6):
                nx = math.cos(math.radians(oangle + 30 + 60 * k))
                ny = math.sin(math.radians(oangle + 30 + 60 * k))
                vals.append(vx * nx + vy * ny)
    vals = np.array(vals)
    m = np.max(vals)
    t_smooth = m + temperature * np.log(np.sum(np.exp((vals - m) / temperature)))
    side = t_smooth * 2.0 / SQRT3
    return side


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        return (x, y)

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0

    # Phase 1: Lattice enumeration with multiple outer orientations
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    oangles_phase1 = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0]
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        for oangle in oangles_phase1:
            if time.time() > deadline:
                break
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            center, side = _compute_outer_for_positions(full_xy, [0.0]*11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle

    if best_positions is None:
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial[:7]]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, 0.0)
        best_oangle = 0.0

    # Phase 2: Spring-mass relaxation from honeycomb topology
    if time.time() < deadline - 20:
        contact_graph = _get_honeycomb_contact_graph()
        init_layout = _build_honeycomb_layout()
        relaxed = _spring_mass_relax(
            [(x, y) for x, y, a in init_layout],
            [a for x, y, a in init_layout],
            contact_graph, n_iter=300, lr=0.005
        )
        rel_pos = [(x, y) for x, y, a in relaxed]
        rel_ang = [a for x, y, a in relaxed]

        for oangle in [0.0, 15.0, -15.0, 30.0, -30.0]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(rel_pos, rel_ang, oangle)
            if side < best_side:
                best_side = side
                best_positions = rel_pos[:]
                best_angles = rel_ang[:]
                best_center = center
                best_oangle = oangle

    # Phase 3: Nonlinear optimization with smooth surrogate
    if time.time() < deadline - 15:
        n_hex = len(best_positions)

        def objective(state):
            # Smooth outer side
            side = _smooth_outer_side(state, n_hex, best_oangle, temperature=0.02)

            # Penalty for overlaps using exact support function check
            angles_list = [state[3*m+2] for m in range(n_hex)]
            penalty = 0.0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    v = _pairwise_violation(state, angles_list, i, j, n_dirs=12)
                    if v > 0:
                        penalty += v ** 2
            return side + 100.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_side + 50:
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

    # Phase 4: Optimize outer angle with current best positions
    if time.time() < deadline - 10:
        for oangle in [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0,
                       -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0, -27.0, -30.0]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle

    # Phase 5: High-precision polish with exact LP objective
    if time.time() < deadline - 5:
        n_hex = len(best_positions)

        def polish_objective(state):
            inner_hexes = [(state[3*i], state[3*i+1], 1.0, state[3*i+2])
                           for i in range(n_hex)]
            c = [0, 0, 1]
            A_ub = []
            b_ub = []
            for cx_h, cy_h, s_h, ang_h in inner_hexes:
                verts = _hex_vertices(cx_h, cy_h, s_h, ang_h)
                for vx, vy in verts:
                    for k in range(6):
                        nx = math.cos(math.radians(best_oangle + 30 + 60 * k))
                        ny = math.sin(math.radians(best_oangle + 30 + 60 * k))
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
                    v = _pairwise_violation(state, angles_list, i, j, n_dirs=12)
                    if v > 0:
                        penalty += v ** 2
            return side + 500.0 * penalty

        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])

        try:
            result = minimize(polish_objective, init_state, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13})
            if result.fun < best_side + 10:
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

    # Final verification
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
