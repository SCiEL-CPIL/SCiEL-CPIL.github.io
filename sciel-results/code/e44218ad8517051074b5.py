"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses smooth surrogate optimization with log-sum-exp for nonsmooth constraints.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE_MARGIN = 15.0
SQRT3 = math.sqrt(3)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _hex_vertices_np(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
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
        return t * 2.0 / SQRT3
    else:
        return float('inf')


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


def _smooth_outer_side_np(state, n_hex, oangle):
    """Vectorized smooth estimate of minimum outer hexagon side using support functions."""
    cx_arr = state[0::3]
    cy_arr = state[1::3]
    ang_arr = state[2::3]
    
    all_vx = np.empty(n_hex * 6)
    all_vy = np.empty(n_hex * 6)
    
    for i in range(n_hex):
        verts = _hex_vertices_np(cx_arr[i], cy_arr[i], 1.0, ang_arr[i])
        all_vx[i*6:(i+1)*6] = verts[:, 0]
        all_vy[i*6:(i+1)*6] = verts[:, 1]
    
    max_s = 0.0
    for k in range(6):
        a_norm = math.radians(oangle + 30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        
        projections = all_vx * nx + all_vy * ny
        range_p = np.max(projections) - np.min(projections)
        s_k = range_p / SQRT3
        if s_k > max_s:
            max_s = s_k
    
    return max_s


def _smooth_pairwise_violation_np(pos, angles, i, j, n_dirs=48, beta=60.0):
    """Smooth approximation of pairwise violation using log-sum-exp with numpy."""
    cx_i = pos[2*i]
    cy_i = pos[2*i+1]
    cx_j = pos[2*j]
    cy_j = pos[2*j+1]
    ang_i = angles[i]
    ang_j = angles[j]
    
    theta = np.radians(360.0 / n_dirs * np.arange(n_dirs))
    nx_arr = np.cos(theta)
    ny_arr = np.sin(theta)
    
    angles_i = np.radians(ang_i + 60 * np.arange(6))
    vx_i = np.cos(angles_i)
    vy_i = np.sin(angles_i)
    
    angles_j = np.radians(ang_j + 60 * np.arange(6))
    vx_j = np.cos(angles_j)
    vy_j = np.sin(angles_j)
    
    proj_i = np.outer(vx_i, nx_arr) + np.outer(vy_i, ny_arr)
    sup_i = np.max(proj_i, axis=0)
    
    proj_j = np.outer(vx_j, -nx_arr) + np.outer(vy_j, -ny_arr)
    sup_j = np.max(proj_j, axis=0)
    
    dx = cx_j - cx_i
    dy = cy_j - cy_i
    dist_proj = dx * nx_arr + dy * ny_arr
    
    violations = sup_i + sup_j - dist_proj
    
    m = np.max(violations)
    if m <= 0:
        return 0.0
    s = np.sum(np.exp(beta * (violations - m)))
    return m + math.log(s) / beta


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
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
        
        for oangle in [0.0, 15.0, 30.0, -15.0, -30.0]:
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
    
    # Nonlinear optimization to improve packing using smooth surrogates
    if time.time() < deadline - 25:
        n_hex = len(best_positions)
        
        def state_to_hexes(state):
            hexes = []
            for i in range(n_hex):
                hexes.append((state[3*i], state[3*i+1], 1.0, state[3*i+2]))
            return hexes
        
        def objective_smooth(state, beta=60.0, n_dirs=48, penalty_weight=50.0):
            """Smooth objective using smooth outer side estimate and violations."""
            angles = [state[3*m+2] for m in range(n_hex)]
            
            side = _smooth_outer_side_np(state, n_hex, best_oangle)
            
            penalty = 0
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    violation = _smooth_pairwise_violation_np(state, angles, i, j, n_dirs=n_dirs, beta=beta)
                    if violation > 0:
                        penalty += violation ** 2
            
            return side + penalty_weight * penalty
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        # Multi-stage optimization with increasing smoothness
        stages = [
            {'beta': 30, 'n_dirs': 36, 'weight': 30, 'maxiter': 150},
            {'beta': 60, 'n_dirs': 48, 'weight': 50, 'maxiter': 200},
            {'beta': 100, 'n_dirs': 60, 'weight': 80, 'maxiter': 250},
            {'beta': 150, 'n_dirs': 72, 'weight': 120, 'maxiter': 300},
        ]
        
        for stage in stages:
            if time.time() > deadline - 10:
                break
                
            def obj_smoothed(state, s=stage):
                angles = [state[3*m+2] for m in range(n_hex)]
                side = _smooth_outer_side_np(state, n_hex, best_oangle)
                penalty = 0
                for i in range(n_hex):
                    for j in range(i + 1, n_hex):
                        violation = _smooth_pairwise_violation_np(state, angles, i, j, n_dirs=s['n_dirs'], beta=s['beta'])
                        if violation > 0:
                            penalty += violation ** 2
                return side + s['weight'] * penalty
            
            try:
                result = minimize(obj_smoothed, init_state, method='L-BFGS-B',
                                 options={'maxiter': stage['maxiter'], 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50})
                if result.fun < best_side + 100:
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
                            init_state = final_state[:]
            except Exception:
                pass
        
        # Nelder-Mead for local refinement
        if time.time() < deadline - 15:
            try:
                def obj_nm(state):
                    angles = [state[3*m+2] for m in range(n_hex)]
                    side = _smooth_outer_side_np(state, n_hex, best_oangle)
                    penalty = 0
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            violation = _smooth_pairwise_violation_np(state, angles, i, j, n_dirs=60, beta=100.0)
                            if violation > 0:
                                penalty += violation ** 2
                    return side + 80.0 * penalty
                
                result_nm = minimize(obj_nm, init_state, method='Nelder-Mead',
                                    options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12})
                if result_nm.fun < best_side + 80:
                    final_state = result_nm.x
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
    
    # Try optimizing outer angle with finer resolution
    if time.time() < deadline - 5:
        oangles_fine = [a / 6.0 for a in range(-60, 61, 1)]
        for oangle in oangles_fine:
            if time.time() > deadline:
                break
            center, side = _compute_outer_for_positions(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
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
        for _ in range(300):
            best_side *= 1.0005
            outer_hex = (best_center[0], best_center[1], best_side, best_oangle)
            if all(_inner_in_outer(h, outer_hex) for h in inner_hexes):
                break
    
    inner = [[inner_hexes[i][0], inner_hexes[i][1], inner_hexes[i][3]] for i in range(len(inner_hexes))]
    
    return inner, (best_center[0], best_center[1]), best_side, best_oangle
