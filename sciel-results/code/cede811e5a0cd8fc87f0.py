"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses smooth nonlinear optimization with scipy to find minimal outer side length.
Exploits hexagonal lattice symmetry: center + ring1 (6) + ring2 (4 selected).
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize, Bounds

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 145.0  # safety margin before 150s limit


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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon containing all hexagons."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / SQRT3


def _axial_to_xy(q, r):
    """Convert axial hex coordinates to Cartesian (flat-top orientation)."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _compute_objective(params, ring2_indices, time_budget):
    """
    Compute the outer hexagon side length given parameters.
    params: [cx, cy, oangle_deg, ring1_scale, ring2_scale]
    - cx, cy: center of the packing
    - oangle_deg: rotation of outer hexagon
    - ring1_scale: radial scale of ring 1 hexagons (1.0 = standard lattice)
    - ring2_scale: radial scale of ring 2 hexagons
    """
    if time.time() > time_budget:
        return 1e10
    
    cx, cy, oangle_deg, r1_scale, r2_scale = params
    
    # Ring 1 positions (6 hexagons around center)
    ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring1_xy = []
    for q, r in ring1_axial:
        x, y = _axial_to_xy(q, r)
        ring1_xy.append((x * r1_scale, y * r1_scale))
    
    # Ring 2 positions (4 selected hexagons)
    ring2_axial = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                   (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                   (0, 2), (1, 2)]
    ring2_xy = []
    for idx in ring2_indices:
        q, r = ring2_axial[idx]
        x, y = _axial_to_xy(q, r)
        ring2_xy.append((x * r2_scale, y * r2_scale))
    
    # All positions relative to center
    all_xy = [(0, 0)] + ring1_xy + ring2_xy
    
    # Build inner hexagons (all with side=1, angle=0)
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in all_xy]
    
    # Compute minimum outer side
    side = _min_outer_side(inner_hexes, cx, cy, oangle_deg)
    
    # Add penalty if hexagons overlap
    penalty = 0.0
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                penalty += 10.0
    
    return side + penalty


def _compute_objective_with_angles(params, ring2_indices, time_budget):
    """
    Extended objective with individual angle optimization for ring 2 hexagons.
    params: [cx, cy, oangle_deg, r1_scale, r2_scale, ang2_0, ang2_1, ang2_2, ang2_3]
    """
    if time.time() > time_budget:
        return 1e10
    
    cx, cy, oangle_deg, r1_scale, r2_scale = params[:5]
    ring2_angles = params[5:9] if len(params) >= 9 else [0.0] * 4
    
    # Ring 1 positions
    ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    ring1_xy = []
    for q, r in ring1_axial:
        x, y = _axial_to_xy(q, r)
        ring1_xy.append((x * r1_scale, y * r1_scale))
    
    # Ring 2 positions
    ring2_axial = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                   (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                   (0, 2), (1, 2)]
    ring2_xy = []
    for idx in ring2_indices:
        q, r = ring2_axial[idx]
        x, y = _axial_to_xy(q, r)
        ring2_xy.append((x * r2_scale, y * r2_scale))
    
    all_xy = [(0, 0)] + ring1_xy + ring2_xy
    
    # Build inner hexagons with angles
    inner_hexes = [(0, 0, 1.0, 0.0)]
    for x, y in ring1_xy:
        inner_hexes.append((x, y, 1.0, 0.0))
    for k, (x, y) in enumerate(ring2_xy):
        inner_hexes.append((x, y, 1.0, ring2_angles[k]))
    
    side = _min_outer_side(inner_hexes, cx, cy, oangle_deg)
    
    penalty = 0.0
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                penalty += 10.0
    
    return side + penalty


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Ring 2 axial coordinates (14 positions)
    ring2_axial = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
                   (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
                   (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_params = None
    best_ring2 = None
    
    # Generate all combinations of 4 ring-2 positions
    all_combos = list(itertools.combinations(range(len(ring2_axial)), 4))
    
    # Phase 1: Coarse scan with scipy optimization for each combination
    for combo in all_combos:
        if time.time() > deadline - 30.0:
            break
        
        ring2_indices = list(combo)
        
        # Initial guess: standard lattice positions
        x0 = [0.0, 0.0, 0.0, 1.0, 1.0]
        
        # Try multiple outer orientations
        for oang_init in [0.0, 30.0, 15.0, -15.0, 45.0]:
            if time.time() > deadline - 30.0:
                break
            
            x0[2] = oang_init
            
            # Bounds: center offset, angle, scales
            bounds = [
                (-2.0, 2.0),   # cx
                (-2.0, 2.0),   # cy
                (-90.0, 90.0), # oangle_deg
                (0.95, 1.05),  # ring1_scale
                (0.95, 1.05),  # ring2_scale
            ]
            
            try:
                result = minimize(
                    _compute_objective,
                    x0,
                    args=(ring2_indices, deadline),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success and result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x.copy()
                    best_ring2 = ring2_indices
                    
            except Exception:
                pass
    
    # Phase 2: Fine-tune with extended parameters (including ring 2 angles)
    if best_params is not None and time.time() < deadline - 15.0:
        # Start from best found
        x0_ext = list(best_params) + [0.0, 0.0, 0.0, 0.0]
        
        bounds_ext = [
            (-2.0, 2.0),   # cx
            (-2.0, 2.0),   # cy
            (-90.0, 90.0), # oangle_deg
            (0.95, 1.05),  # ring1_scale
            (0.95, 1.05),  # ring2_scale
            (-30.0, 30.0), # ring2 angle 0
            (-30.0, 30.0), # ring2 angle 1
            (-30.0, 30.0), # ring2 angle 2
            (-30.0, 30.0), # ring2 angle 3
        ]
        
        try:
            result = minimize(
                _compute_objective_with_angles,
                x0_ext,
                args=(best_ring2, deadline),
                method='L-BFGS-B',
                bounds=bounds_ext,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    # Phase 3: High-precision polish
    if best_params is not None and time.time() < deadline - 5.0:
        x0_final = best_params.copy()
        
        bounds_final = [
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-90.0, 90.0),
            (0.98, 1.02),
            (0.98, 1.02),
        ] + [(-15.0, 15.0)] * 4
        
        try:
            result = minimize(
                _compute_objective_with_angles,
                x0_final,
                args=(best_ring2, deadline),
                method='L-BFGS-B',
                bounds=bounds_final,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success and result.fun < best_side:
                best_side = result.fun
                best_params = result.x.copy()
                
        except Exception:
            pass
    
    # Build final configuration from best parameters
    if best_params is None:
        # Fallback: standard lattice
        ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        ring2_axial_subset = [(2, 0), (2, -1), (2, -2), (1, -2)]
        all_axial = [(0, 0)] + ring1_axial + ring2_axial_subset
        all_xy = [_axial_to_xy(q, r) for q, r in all_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in all_xy]
        cx_mean = sum(x for x, y in all_xy) / 11.0
        cy_mean = sum(y for x, y in all_xy) / 11.0
        best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
        best_center = (cx_mean, cy_mean)
        best_angle = 0.0
    else:
        cx, cy, oangle_deg, r1_scale, r2_scale = best_params[:5]
        ring2_angles = best_params[5:9] if len(best_params) >= 9 else [0.0] * 4
        
        ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        ring1_xy = []
        for q, r in ring1_axial:
            x, y = _axial_to_xy(q, r)
            ring1_xy.append((x * r1_scale, y * r1_scale))
        
        ring2_xy = []
        for idx in best_ring2:
            q, r = ring2_axial[idx]
            x, y = _axial_to_xy(q, r)
            ring2_xy.append((x * r2_scale, y * r2_scale))
        
        all_xy = [(0, 0)] + ring1_xy + ring2_xy
        
        inner_hexes = [(0, 0, 1.0, 0.0)]
        for x, y in ring1_xy:
            inner_hexes.append((x, y, 1.0, 0.0))
        for k, (x, y) in enumerate(ring2_xy):
            inner_hexes.append((x, y, 1.0, ring2_angles[k]))
        
        best_center = (cx, cy)
        best_angle = oangle_deg
    
    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Try to fix by scaling positions slightly
        for scale in [1.001, 1.005, 1.01, 1.02]:
            scaled_hexes = [(x * scale, y * scale, s, a) for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, best_angle)
                best_center = (cx_mean, cy_mean)
                break
    
    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
    
    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    
    return inner, (best_center[0], best_center[1]), best_side, best_angle
