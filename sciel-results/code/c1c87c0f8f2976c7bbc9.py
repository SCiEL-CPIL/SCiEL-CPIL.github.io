"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses convex optimization (LP for outer hexagon, nonlinear for inner positions).
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit


def _softmax(values, beta=50.0):
    """Smooth max via log-sum-exp."""
    m = max(values)
    if m == 0:
        return 0.0
    s = sum(math.exp(beta * (v - m)) for v in values)
    return m + math.log(s) / beta


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
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
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
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


def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
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
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        t = res.x[2]
        return t * 2.0 / math.sqrt(3)
    else:
        return float('inf')


def _outer_side_smooth(pos, angles, oangle, beta=100.0):
    """Smooth outer side length using support functions and log-sum-exp.
    
    The outer hexagon with center (cx,cy), side s, angle oangle contains
    a point (px,py) iff for all 6 face normals: (px-cx)*nx + (py-cy)*ny <= apothem
    where apothem = s*sqrt(3)/2.
    
    So s >= 2/sqrt(3) * max over all points and all faces of [(px-cx)*nx + (py-cy)*ny]
    We minimize over (cx,cy) which gives:
    s >= 2/sqrt(3) * max_face [max_point support - min_point support] / 2 ... 
    Actually simpler: the minimal enclosing hexagon side is determined by the
    maximum extent in each of the 6 face normal directions.
    """
    sqrt3 = math.sqrt(3)
    n_hex = len(angles)
    
    # For each of the 6 face directions, compute max and min projection of all vertices
    # The width in direction k is max_proj - min_proj
    # side >= width_k / sqrt(3) for all k
    # So side >= max_k(width_k / sqrt(3))
    
    # We want to minimize over center (cx, cy)
    # width_k = max_p(p_k) - min_p(p_k) where p_k is projection onto normal k
    # This is independent of center! So the minimal side is just max_k(width_k/sqrt(3))
    
    # But we also need to return the center. The optimal center is the midpoint of
    # the min and max projections in each direction, but they must be consistent.
    # For now, compute the side and use centroid as center.
    
    max_width = 0.0
    for k in range(6):
        a_norm = math.radians(oangle + 30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        
        min_proj = float('inf')
        max_proj = float('-inf')
        
        for i in range(n_hex):
            cx_h = pos[2*i]
            cy_h = pos[2*i+1]
            ang_h = angles[i]
            verts = _hex_vertices(cx_h, cy_h, 1.0, ang_h)
            for vx, vy in verts:
                proj = vx * nx + vy * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
        
        width = max_proj - min_proj
        if width > max_width:
            max_width = width
    
    side = max_width / sqrt3
    return side


def _compute_outer_for_positions(positions, angles, oangle=0.0):
    """Compute optimal outer hexagon for given inner positions and angles."""
    inner_hexes = [(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(len(positions))]
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
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
    
    if res.success:
        center = (res.x[0], res.x[1])
    else:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    
    return center, side


def _compute_outer_center_and_side(positions, angles, oangle=0.0):
    """Compute outer hexagon center and side using width-based method (fast, no LP)."""
    sqrt3 = math.sqrt(3)
    n_hex = len(angles)
    
    # Compute width in each of 6 face-normal directions
    widths = []
    min_projs = []
    max_projs = []
    
    for k in range(6):
        a_norm = math.radians(oangle + 30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        
        min_proj = float('inf')
        max_proj = float('-inf')
        
        for i in range(n_hex):
            cx_h = positions[i][0]
            cy_h = positions[i][1]
            ang_h = angles[i]
            verts = _hex_vertices(cx_h, cy_h, 1.0, ang_h)
            for vx, vy in verts:
                proj = vx * nx + vy * ny
                if proj < min_proj:
                    min_proj = proj
                if proj > max_proj:
                    max_proj = proj
        
        width = max_proj - min_proj
        widths.append(width)
        min_projs.append(min_proj)
        max_projs.append(max_proj)
    
    side = max(widths) / sqrt3
    
    # Center: the midpoint that works for all directions
    # For direction k, center projection should be (min_proj_k + max_proj_k) / 2
    # We solve for (cx, cy) that best satisfies all 6 constraints
    # cx*nx_k + cy*ny_k = (min_proj_k + max_proj_k)/2 for k=0..5
    # This is an overdetermined system; use least squares
    A = []
    b = []
    for k in range(6):
        a_norm = math.radians(oangle + 30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        A.append([nx, ny])
        b.append((min_projs[k] + max_projs[k]) / 2.0)
    
    A_arr = np.array(A)
    b_arr = np.array(b)
    try:
        center_arr, _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=None)
        center = (float(center_arr[0]), float(center_arr[1]))
    except:
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        center = (cx_mean, cy_mean)
    
    return center, side


def _support_function(cx, cy, s, ang, nx, ny):
    """Support function of hexagon in direction (nx, ny)."""
    max_val = 0
    for m in range(6):
        a = math.radians(ang + 60 * m)
        vx = s * math.cos(a)
        vy = s * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _pairwise_violation(pos, angles, i, j):
    """Check if hexagons i and j are disjoint. Return violation (positive = overlap)."""
    max_violation = 0
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


def _smooth_pairwise_violation(pos, angles, i, j, beta=40.0):
    """Smooth approximation of pairwise violation using log-sum-exp."""
    violations = []
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
        violations.append(violation)
    
    return _softmax(violations, beta)


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    sqrt3 = math.sqrt(3)
    
    # Initial lattice configuration
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    # Try combinations of 4 from ring2 with various outer angles
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        for oangle in [0.0, 7.5, 15.0, 22.5, 30.0, -7.5, -15.0, -22.5, -30.0]:
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
    
    # Nonlinear optimization to improve packing
    if time.time() < deadline:
        n_hex = len(best_positions)
        
        def state_to_hexes(state):
            hexes = []
            for i in range(n_hex):
                hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))
            return hexes
        
        def objective_smooth(state):
            """Smooth objective using log-sum-exp for both outer side and violations."""
            angles = [state[3*m+2] for m in range(n_hex)]
            positions = [(state[2*i], state[2*i+1]) for i in range(n_hex)]
            
            # Compute outer side via width method (fast, no LP)
            side = _outer_side_smooth(state, angles, best_oangle)
            
            # Smooth penalty for pairwise violations
            penalty = 0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    violation = _smooth_pairwise_violation(state, angles, i, j, beta=40.0)
                    if violation > 0:
                        penalty += violation ** 2
            
            return side + 50.0 * penalty
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        try:
            result = minimize(objective_smooth, init_state, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11})
            if result.fun < best_side + 50:
                final_state = result.x
                hexes = state_to_hexes(final_state)
                
                if _all_disjoint(hexes):
                    inner_hexes = [(h[0], h[1], h[2], h[3]) for h in hexes]
                    center, side = _compute_outer_for_positions(
                        [(h[0], h[1]) for h in hexes], 
                        [h[3] for h in hexes], 
                        best_oangle
                    )
                    if side < best_side:
                        best_side = side
                        best_positions = [(h[0], h[1]) for h in hexes]
                        best_angles = [h[3] for h in hexes]
                        best_center = center
        except Exception:
            pass
        
        # High-precision polish pass
        if time.time() < deadline - 15:
            try:
                angles_polish = [best_angles[m] for m in range(n_hex)]
                
                def objective_polish(state):
                    angles_p = [state[3*m+2] for m in range(n_hex)]
                    
                    # Fast width-based side computation
                    side = _outer_side_smooth(state, angles_p, best_oangle)
                    
                    penalty = 0
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            violation = _smooth_pairwise_violation(state, angles_p, i, j, beta=80.0)
                            if violation > 0:
                                penalty += violation ** 2
                    
                    return side + 100.0 * penalty
                
                init_polish = []
                for i in range(n_hex):
                    init_polish.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
                
                result_p = minimize(objective_polish, init_polish, method='Nelder-Mead', 
                                   options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13})
                if result_p.fun < best_side + 100:
                    final_state_p = result_p.x
                    hexes_p = state_to_hexes(final_state_p)
                    
                    if _all_disjoint(hexes_p):
                        inner_hexes_p = [(h[0], h[1], h[2], h[3]) for h in hexes_p]
                        center_p, side_p = _compute_outer_for_positions(
                            [(h[0], h[1]) for h in hexes_p], 
                            [h[3] for h in hexes_p], 
                            best_oangle
                        )
                        if side_p < best_side:
                            best_side = side_p
                            best_positions = [(h[0], h[1]) for h in hexes_p]
                            best_angles = [h[3] for h in hexes_p]
                            best_center = center_p
            except Exception:
                pass
            
            # Second polish with even tighter tolerances if time allows
            if time.time() < deadline - 5:
                try:
                    def objective_polish2(state):
                        angles_p2 = [state[3*m+2] for m in range(n_hex)]
                        side2 = _outer_side_smooth(state, angles_p2, best_oangle)
                        penalty2 = 0
                        for i in range(n_hex):
                            for j in range(i + 1, n_hex):
                                violation2 = _smooth_pairwise_violation(state, angles_p2, i, j, beta=120.0)
                                if violation2 > 0:
                                    penalty2 += violation2 ** 2
                        return side2 + 200.0 * penalty2
                    
                    init_polish2 = []
                    for i in range(n_hex):
                        init_polish2.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
                    
                    result_p2 = minimize(objective_polish2, init_polish2, method='Nelder-Mead', 
                                       options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if result_p2.fun < best_side + 200:
                        final_state_p2 = result_p2.x
                        hexes_p2 = state_to_hexes(final_state_p2)
                        
                        if _all_disjoint(hexes_p2):
                            center_p2, side_p2 = _compute_outer_for_positions(
                                [(h[0], h[1]) for h in hexes_p2], 
                                [h[3] for h in hexes_p2], 
                                best_oangle
                            )
                            if side_p2 < best_side:
                                best_side = side_p2
                                best_positions = [(h[0], h[1]) for h in hexes_p2]
                                best_angles = [h[3] for h in hexes_p2]
                                best_center = center_p2
                except Exception:
                    pass
    
    # Try optimizing outer angle with finer granularity
    if time.time() < deadline - 20:
        # First pass: coarse
        for oangle in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
        
        # Second pass: fine around best
        if time.time() < deadline - 10:
            for da in [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]:
                if time.time() > deadline:
                    break
                oangle_fine = best_oangle + da
                center, side = _compute_outer_for_positions(best_positions, best_angles, oangle_fine)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_oangle = oangle_fine
    
    # Final verification
    inner_hexes = [(best_positions[i][0], best_positions[i][1], 1.0, best_angles[i]) for i in range(len(best_positions))]
    
    if not _all_disjoint(inner_hexes):
        for scale in [1.001, 1.005, 1.01, 1.02]:
            scaled_hexes = [(x * scale, y * scale, 1.0, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                best_positions = [(h[0], h[1]) for h in inner_hexes]
                best_angles = [h[3] for h in inner_hexes]
                best_center, best_side = _compute_outer_for_positions(best_positions, best_angles, best_oangle)
                break
    
    outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        for _ in range(200):
            best_side *= 1.0005
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break
    
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
