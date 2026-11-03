"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses exact support functions and L-BFGS-B with smooth objectives.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
DEADLINE = 140.0  # seconds, safety margin before 150s limit


def _support_hex(side, angle_deg, nx, ny):
    """Support function of a regular hexagon (side, angle) in direction (nx,ny)."""
    max_val = 0.0
    a_rad = math.radians(angle_deg)
    for k in range(6):
        a = a_rad + math.radians(60 * k)
        vx = side * math.cos(a)
        vy = side * math.sin(a)
        val = vx * nx + vy * ny
        if val > max_val:
            max_val = val
    return max_val


def _overlap_violation(cx1, cy1, a1, cx2, cy2, a2):
    """
    Exact overlap violation between two unit hexagons.
    Returns max over critical directions of [support(H1⊕H2, d) - (c2-c1)·d].
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist2 = dx * dx + dy * dy
    if dist2 > 4.0:
        return 0.0
    
    max_viol = 0.0
    for k in range(6):
        for a_base in [a1, a2]:
            theta = math.radians(a_base + 30 + 60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            s = _support_hex(1.0, a1, nx, ny) + _support_hex(1.0, a2, nx, ny)
            proj = dx * nx + dy * ny
            viol = s - proj
            if viol > max_viol:
                max_viol = viol
    return max_viol


def _all_overlap_penalty(positions, angles, weight=100.0):
    """Compute total overlap penalty for all pairs."""
    n = len(positions)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            viol = _overlap_violation(
                positions[i][0], positions[i][1], angles[i],
                positions[j][0], positions[j][1], angles[j]
            )
            if viol > 0:
                penalty += viol * viol
    return weight * penalty


def _outer_side_lp(positions, angles, oangle_deg):
    """Find minimum side length of outer hexagon via LP."""
    n = len(positions)
    c = [0, 0, 1]
    A_ub = []
    b_ub = []
    
    oangle_rad = math.radians(oangle_deg)
    normals = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    for i in range(n):
        cx_h, cy_h = positions[i]
        ang_h = angles[i]
        for k in range(6):
            a = math.radians(ang_h + 60 * k)
            vx = cx_h + math.cos(a)
            vy = cy_h + math.sin(a)
            for nx, ny in normals:
                A_ub.append([-nx, -ny, -1])
                b_ub.append(-vx * nx - vy * ny)
    
    A_ub.append([0, 0, -1])
    b_ub.append(0)
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None), (None, None), (0, None)],
                  method='highs')
    if not res.success:
        return float('inf'), (0, 0)
    t = res.x[2]
    side = t * 2.0 / math.sqrt(3)
    center = (res.x[0], res.x[1])
    return side, center


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


def _inner_in_outer(cx, cy, ang, outer_cx, outer_cy, outer_side, outer_ang):
    """Check if inner hexagon is fully inside outer hexagon."""
    for k in range(6):
        a = math.radians(ang + 60 * k)
        vx = cx + math.cos(a)
        vy = cy + math.sin(a)
        if not _point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_ang):
            return False
    return True


def _all_disjoint(positions, angles, eps=1e-9):
    """Check if all pairs are disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            viol = _overlap_violation(
                positions[i][0], positions[i][1], angles[i],
                positions[j][0], positions[j][1], angles[j]
            )
            if viol > eps:
                return False
    return True


def _axial_to_xy(q, r):
    """Convert axial hex coordinates to Cartesian."""
    sqrt3 = math.sqrt(3)
    x = 1.5 * q
    y = sqrt3 * (r + q / 2.0)
    return (x, y)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    sqrt3 = math.sqrt(3)
    
    # Initial 1-2-3-2-1 configuration
    base_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring2_full = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_oangle = 0.0
    
    # Generate promising 4-subsets of ring2
    promising_subsets = []
    # Contiguous arcs of length 4
    for start in range(12):
        subset = tuple(sorted([(start + k) % 12 for k in range(4)]))
        if subset not in promising_subsets:
            promising_subsets.append(subset)
    # Symmetric: two opposite pairs
    for i in range(6):
        for k in range(i + 1, 6):
            subset = tuple(sorted([i, (i + 6) % 12, k, (k + 6) % 12]))
            if len(set(subset)) == 4 and subset not in promising_subsets:
                promising_subsets.append(subset)
    
    oangles_to_try = [0.0, 15.0, 30.0, -15.0, -30.0, 45.0, -45.0, 10.0, -10.0, 20.0, -20.0, 5.0, -5.0]
    
    # Phase 1: Evaluate configurations with LP
    for subset in promising_subsets:
        if time.time() > deadline - 25:
            break
        
        selected_r2 = [ring2_full[i] for i in subset]
        full_axial = base_axial + ring1 + selected_r2
        full_xy = [_axial_to_xy(q, r) for q, r in full_axial]
        
        for oangle in oangles_to_try:
            if time.time() > deadline - 25:
                break
            side, center = _outer_side_lp(full_xy, [0.0] * 11, oangle)
            if side < best_side:
                best_side = side
                best_positions = full_xy[:]
                best_angles = [0.0] * 11
                best_center = center
                best_oangle = oangle
    
    if best_positions is None:
        full_xy = [_axial_to_xy(q, r) for q, r in base_axial + ring1]
        best_positions = full_xy
        best_angles = [0.0] * 7
        best_side, best_center = _outer_side_lp(best_positions, best_angles, 0.0)
        best_oangle = 0.0
    
    # Phase 2: L-BFGS-B optimization of positions and angles
    if time.time() < deadline - 8:
        n_hex = len(best_positions)
        
        def objective(state):
            positions = [(state[3*i], state[3*i+1]) for i in range(n_hex)]
            angles = [state[3*i+2] for i in range(n_hex)]
            
            side, center = _outer_side_lp(positions, angles, best_oangle)
            if side > 100:
                return 1e6
            
            penalty = _all_overlap_penalty(positions, angles, weight=50.0)
            return side + penalty
        
        init_state = []
        for i in range(n_hex):
            init_state.extend([best_positions[i][0], best_positions[i][1], best_angles[i]])
        
        try:
            result = minimize(
                objective, init_state, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            final_state = result.x
            positions = [(final_state[3*i], final_state[3*i+1]) for i in range(n_hex)]
            angles = [final_state[3*i+2] for i in range(n_hex)]
            
            if _all_disjoint(positions, angles, eps=1e-7):
                side, center = _outer_side_lp(positions, angles, best_oangle)
                if side < best_side:
                    best_side = side
                    best_positions = positions
                    best_angles = angles
                    best_center = center
        except Exception:
            pass
    
    # Phase 3: Try more outer angles
    if time.time() < deadline - 4:
        for oangle in [best_oangle + d for d in [-10, -5, 0, 5, 10]] + [0, 15, 30, -15, -30]:
            if time.time() > deadline - 2:
                break
            side, center = _outer_side_lp(best_positions, best_angles, oangle)
            if side < best_side:
                best_side = side
                best_center = center
                best_oangle = oangle
    
    # Phase 4: Final polish
    if time.time() < deadline - 1:
        for scale in [1.0001, 1.0005, 1.001]:
            new_positions = [(x * scale, y * scale) for x, y in best_positions]
            if _all_disjoint(new_positions, best_angles, eps=1e-8):
                side, center = _outer_side_lp(new_positions, best_angles, best_oangle)
                if side < best_side:
                    best_side = side
                    best_positions = new_positions
                    best_center = center
    
    # Final: ensure containment
    outer_cx, outer_cy = best_center
    outer_side = best_side
    outer_ang = best_oangle
    
    for _ in range(100):
        valid = all(
            _inner_in_outer(best_positions[i][0], best_positions[i][1], best_angles[i],
                           outer_cx, outer_cy, outer_side, outer_ang)
            for i in range(len(best_positions))
        )
        if valid:
            break
        outer_side *= 1.001
    best_side = outer_side
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(len(best_positions))]
    
    return inner, (outer_cx, outer_cy), best_side, best_oangle
