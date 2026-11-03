"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses LP-based exact outer hexagon computation combined with smooth
nonlinear optimization of inner positions. Exploits hexagonal symmetry
with center + ring1 (6) + ring2 (4 selected) structure.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 148.0  # safety margin before 150s limit


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


def _min_outer_side_lp(inner_hexes):
    """
    Find minimum side length of outer hexagon (free center and angle)
    containing all inner hexagons using LP.
    
    The outer hexagon is defined by 6 half-planes. For a regular hexagon
    with side S, center (ox, oy), angle theta, the constraint for vertex v
    of each inner hexagon is:
        For each face direction k: n_k . (v - c) <= S * sqrt(3)/2
    where n_k are the 6 outward normals.
    
    We minimize S over (ox, oy, theta, S).
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, s, a in inner_hexes:
        verts = _hex_vertices(cx, cy, s, a)
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0, (0.0, 0.0, 0.0)
    
    # For a given angle theta, the outer hexagon with center (ox, oy) and side S
    # contains point (px, py) iff for all k:
    #   n_k(theta) . ((px, py) - (ox, oy)) <= S * sqrt(3)/2
    # where n_k(theta) = (cos(30+60k+theta), sin(30+60k+theta))
    #
    # This is equivalent to:
    #   n_k . (px, py) - n_k . (ox, oy) <= S * sqrt(3)/2
    #   n_k . (ox, oy) >= n_k . (px, py) - S * sqrt(3)/2
    #
    # And also for the opposite direction:
    #   -n_k . ((px, py) - (ox, oy)) <= S * sqrt(3)/2
    #   -n_k . (px, py) + n_k . (ox, oy) <= S * sqrt(3)/2
    #   n_k . (ox, oy) <= n_k . (px, py) + S * sqrt(3)/2
    #
    # So: max_k (n_k . v) - min_k (n_k . v) <= S * sqrt(3)  for each v
    # Wait, that's not quite right. Let me think more carefully.
    #
    # A regular hexagon with side S, center c, angle theta has the property that
    # a point p is inside iff for all 6 face normals n_k:
    #   n_k . (p - c) <= S * sqrt(3) / 2
    #
    # So for ALL vertices v of all inner hexagons:
    #   n_k(theta) . v - n_k(theta) . c <= S * sqrt(3) / 2
    #
    # This means:
    #   n_k(theta) . c >= n_k(theta) . v - S * sqrt(3) / 2   for all v, k
    #   n_k(theta) . c <= n_k(theta) . v + S * sqrt(3) / 2   for all v, k (using -n_k)
    #
    # Actually, the 6 face normals are n_k and -n_k for k=0,1,2 (since they come in pairs).
    # Let me use all 6 normals n_k for k=0..5.
    #
    # Constraint: for all k, v:
    #   n_k . v - n_k . c <= S * sqrt(3)/2
    #   -n_k . v + n_k . c <= S * sqrt(3)/2
    #
    # Equivalently: |n_k . (v - c)| <= S * sqrt(3)/2 for all k, v
    #
    # So: n_k . c >= max_v(n_k . v) - S*sqrt(3)/2
    #     n_k . c <= min_v(n_k . v) + S*sqrt(3)/2
    #
    # For feasibility: max_v(n_k . v) - S*sqrt(3)/2 <= min_v(n_k . v) + S*sqrt(3)/2
    # i.e., max_v(n_k . v) - min_v(n_k . v) <= S * sqrt(3)
    #
    # So S >= (max_v(n_k . v) - min_v(n_k . v)) / sqrt(3) for all k
    #
    # And c must satisfy: max_v(n_k . v) - S*sqrt(3)/2 <= n_k . c <= min_v(n_k . v) + S*sqrt(3)/2
    #
    # The minimum S for a given theta is:
    #   S_min(theta) = max_k [ (max_v(n_k(theta).v) - min_v(n_k(theta).v)) / sqrt(3) ]
    #
    # And we need to check that there exists a center c satisfying all constraints.
    # The center c is feasible if the intersection of all the strips is non-empty.
    # For a regular hexagon, this is automatically satisfied if S >= S_min.
    #
    # Actually, let me just do a 2D optimization over (ox, oy) for each theta,
    # or better, optimize over (ox, oy, theta) jointly.
    
    # Simpler approach: optimize over theta (0 to 60 degrees by symmetry),
    # and for each theta, find the best (ox, oy) and S.
    
    # For fixed theta, the problem reduces to finding the smallest S and center (ox, oy)
    # such that all points are in the hexagon. This is a linear program in (ox, oy, S).
    # Variables: ox, oy, S
    # Constraints: for all k, v:
    #   n_k . (v - (ox,oy)) <= S * sqrt(3)/2
    #   -n_k . (v - (ox,oy)) <= S * sqrt(3)/2
    #   S >= 0
    
    # Let's use scipy's linprog or just do a grid search over theta + LP for (ox, oy, S)
    
    # Actually, let me use a different approach. For fixed theta, I can compute
    # S_min and check feasibility of center.
    
    # Let me just do a smooth optimization over (ox, oy, theta) with the objective
    # being the minimum S needed.
    
    # For a given (ox, oy, theta), the required S is:
    #   S = max over all k, v of |n_k(theta) . (v - (ox, oy))| / (sqrt(3)/2)
    #     = max over all k, v of |n_k(theta) . (v - (ox, oy))| * 2 / sqrt(3)
    
    # This is a convex function of (ox, oy) for fixed theta, but non-smooth.
    # Let me just return the best I can find with a simple approach.
    
    # Quick approach: try several theta values and for each, find optimal center
    # via the formula, then return the best.
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_theta = 0.0
    
    # By symmetry, theta in [0, 30] degrees is sufficient
    for theta_deg in range(0, 31, 1):
        theta = math.radians(theta_deg)
        normals = []
        for k in range(6):
            a = math.radians(30 + 60 * k) + theta
            normals.append((math.cos(a), math.sin(a)))
        
        # For each normal, compute max and min projection of all vertices
        max_projs = []
        min_projs = []
        for nx, ny in normals:
            projs = [nx * vx + ny * vy for vx, vy in all_verts]
            max_projs.append(max(projs))
            min_projs.append(min(projs))
        
        # Required S for this theta (ignoring center optimization for now,
        # using center at origin)
        # Actually, we can optimize center: the optimal center for a given theta
        # and S is the centroid of the feasible region. But for the minimum S,
        # we need: for all k, max_v(n_k.v) - min_v(n_k.v) <= S*sqrt(3)
        # The minimum S is max_k (max_v(n_k.v) - min_v(n_k.v)) / sqrt(3)
        # But this assumes center can be chosen to center the projections.
        # The optimal center satisfies: n_k . c = (max_v(n_k.v) + min_v(n_k.v)) / 2
        # for all k. This is only consistent if these values agree.
        
        # For a regular hexagon, the 6 normals are not independent (only 3 independent).
        # Let me just compute S assuming center at the average of all vertices
        # for each theta, then refine.
        
        cx_avg = sum(vx for vx, vy in all_verts) / len(all_verts)
        cy_avg = sum(vy for vx, vy in all_verts) / len(all_verts)
        
        S_req = 0.0
        for k, (nx, ny) in enumerate(normals):
            for vx, vy in all_verts:
                proj = abs(nx * (vx - cx_avg) + ny * (vy - cy_avg))
                if proj > S_req:
                    S_req = proj
        S_req *= 2.0 / SQRT3
        
        if S_req < best_S:
            best_S = S_req
            best_center = (cx_avg, cy_avg)
            best_theta = theta_deg
    
    # Refine with finer theta search around best
    for _ in range(3):
        theta_coarse = best_theta
        best_S_local = float('inf')
        best_center_local = best_center
        best_theta_local = best_theta
        
        for i in range(-5, 6):
            theta_deg = theta_coarse + i * 0.5
            if theta_deg < 0 or theta_deg > 30:
                continue
            theta = math.radians(theta_deg)
            normals = []
            for k in range(6):
                a = math.radians(30 + 60 * k) + theta
                normals.append((math.cos(a), math.sin(a)))
            
            cx_avg = sum(vx for vx, vy in all_verts) / len(all_verts)
            cy_avg = sum(vy for vx, vy in all_verts) / len(all_verts)
            
            S_req = 0.0
            for k, (nx, ny) in enumerate(normals):
                for vx, vy in all_verts:
                    proj = abs(nx * (vx - cx_avg) + ny * (vy - cy_avg))
                    if proj > S_req:
                        S_req = proj
            S_req *= 2.0 / SQRT3
            
            if S_req < best_S_local:
                best_S_local = S_req
                best_center_local = (cx_avg, cy_avg)
                best_theta_local = theta_deg
        
        if best_S_local >= best_S:
            break
        best_S = best_S_local
        best_center = best_center_local
        best_theta = best_theta_local
    
    return best_S, (best_center[0], best_center[1], best_theta)


def _min_outer_side_fixed(inner_hexes, ox, oy, oangle_deg):
    """Find minimum side length of outer hexagon with fixed center and angle."""
    max_val = 0.0
    a = -math.radians(oangle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in inner_hexes:
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


def _compute_objective(params, ring2_indices, deadline, use_lp=False):
    """
    Compute the outer hexagon side length given parameters.
    params: [cx, cy, oangle_deg, ring1_scale, ring2_scale]
    """
    if time.time() > deadline:
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
    
    if use_lp:
        side, (ox, oy, oa) = _min_outer_side_lp(inner_hexes)
    else:
        side = _min_outer_side_fixed(inner_hexes, cx, cy, oangle_deg)
    
    # Add penalty if hexagons overlap
    penalty = 0.0
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                penalty += 10.0
    
    return side + penalty


def _compute_objective_full(params, ring2_indices, deadline):
    """
    Full objective with individual position and angle for each hexagon.
    params: [x0,y0, x1,y1,a1, x2,y2,a2, ..., x10,y10,a10]  (33 params)
    """
    if time.time() > deadline:
        return 1e10
    
    n = len(params) // 3
    inner_hexes = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, 1.0, a))
    
    # Compute outer side using LP (free center and angle)
    side, (ox, oy, oa) = _min_outer_side_lp(inner_hexes)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                penalty += 5.0
    
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
    best_outer = None  # (ox, oy, oangle)
    
    # Generate all combinations of 4 ring-2 positions
    all_combos = list(itertools.combinations(range(len(ring2_axial)), 4))
    
    # Phase 1: Coarse scan with scipy optimization for each combination
    # Use LP-based outer for better results
    for combo in all_combos:
        if time.time() > deadline - 40.0:
            break
        
        ring2_indices = list(combo)
        
        # Initial guess
        x0 = [0.0, 0.0, 0.0, 1.0, 1.0]
        
        # Try multiple outer orientations
        for oang_init in [0.0, 30.0, 15.0, -15.0, 45.0, 75.0]:
            if time.time() > deadline - 40.0:
                break
            
            x0[2] = oang_init
            
            bounds = [
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-90.0, 90.0),
                (0.95, 1.05),
                (0.95, 1.05),
            ]
            
            try:
                result = minimize(
                    _compute_objective,
                    x0,
                    args=(ring2_indices, deadline, False),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success and result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x.copy()
                    best_ring2 = ring2_indices
                    best_outer = (result.x[0], result.x[1], result.x[2])
                    
            except Exception:
                pass
    
    # Phase 2: Refine with LP-based outer hexagon
    if best_params is not None and time.time() < deadline - 25.0:
        for combo_try in [best_ring2] if best_ring2 else []:
            if time.time() > deadline - 25.0:
                break
            
            ring2_indices = list(combo_try)
            x0 = best_params.copy()
            
            bounds = [
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-90.0, 90.0),
                (0.96, 1.04),
                (0.96, 1.04),
            ]
            
            try:
                result = minimize(
                    _compute_objective,
                    x0,
                    args=(ring2_indices, deadline, True),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x.copy()
                    best_ring2 = ring2_indices
                    # Get outer from LP
                    ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
                    r1_scale = result.x[3]
                    r2_scale = result.x[4]
                    ring1_xy = []
                    for q, r in ring1_axial:
                        x, y = _axial_to_xy(q, r)
                        ring1_xy.append((x * r1_scale, y * r1_scale))
                    ring2_xy = []
                    for idx in ring2_indices:
                        q, r = ring2_axial[idx]
                        x, y = _axial_to_xy(q, r)
                        ring2_xy.append((x * r2_scale, y * r2_scale))
                    all_xy = [(0, 0)] + ring1_xy + ring2_xy
                    inner_hexes = [(x, y, 1.0, 0.0) for x, y in all_xy]
                    _, best_outer = _min_outer_side_lp(inner_hexes)
                    
            except Exception:
                pass
    
    # Phase 3: Full free optimization of all 11 hexagon positions and angles
    if time.time() < deadline - 15.0:
        # Initialize from current best
        if best_params is not None and best_ring2 is not None:
            cx, cy, oangle_deg, r1_scale, r2_scale = best_params[:5]
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
            # Full params: x, y, angle for each of 11 hexagons
            x0_full = []
            for x, y in all_xy:
                x0_full.extend([x, y, 0.0])
        else:
            # Fallback initialization
            ring1_axial = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            ring2_subset = [(2, 0), (2, -1), (2, -2), (1, -2)]
            all_axial = [(0, 0)] + ring1_axial + ring2_subset
            all_xy = [_axial_to_xy(q, r) for q, r in all_axial]
            x0_full = []
            for x, y in all_xy:
                x0_full.extend([x, y, 0.0])
        
        # Bounds for full optimization
        bounds_full = []
        for i in range(11):
            bounds_full.extend([(-4.0, 4.0), (-4.0, 4.0), (-30.0, 30.0)])
        
        try:
            result = minimize(
                _compute_objective_full,
                x0_full,
                args=(None, deadline),
                method='L-BFGS-B',
                bounds=bounds_full,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_side:
                best_side = result.fun
                # Extract positions
                inner_hexes_full = []
                for i in range(11):
                    x = result.x[3*i]
                    y = result.x[3*i+1]
                    a = result.x[3*i+2]
                    inner_hexes_full.append((x, y, 1.0, a))
                
                # Get outer
                _, best_outer = _min_outer_side_lp(inner_hexes_full)
                
                # Store for final output
                best_params = result.x.copy()
                best_ring2 = None  # Mark as full optimization
                
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time.time() < deadline - 5.0 and best_params is not None:
        if best_ring2 is None and len(best_params) == 33:
            # Full optimization result - do one more polish
            try:
                result = minimize(
                    _compute_objective_full,
                    best_params,
                    args=(None, deadline),
                    method='L-BFGS-B',
                    bounds=bounds_full,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if result.fun < best_side:
                    best_side = result.fun
                    best_params = result.x.copy()
                    inner_hexes_full = []
                    for i in range(11):
                        x = result.x[3*i]
                        y = result.x[3*i+1]
                        a = result.x[3*i+2]
                        inner_hexes_full.append((x, y, 1.0, a))
                    _, best_outer = _min_outer_side_lp(inner_hexes_full)
            except Exception:
                pass
    
    # Build final configuration
    if best_ring2 is not None:
        # Lattice-based result
        cx, cy, oangle_deg, r1_scale, r2_scale = best_params[:5]
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
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in all_xy]
    else:
        # Full optimization result
        inner_hexes = []
        for i in range(11):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner_hexes.append((x, y, 1.0, a))
    
    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Try to fix by scaling positions slightly
        for scale in [1.001, 1.005, 1.01, 1.02]:
            cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
            cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
            scaled_hexes = [((x - cx_mean) * scale + cx_mean,
                            (y - cy_mean) * scale + cy_mean, s, a)
                           for x, y, s, a in inner_hexes]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                break
    
    # Get outer hexagon
    if best_outer is not None:
        ox, oy, oangle = best_outer
    else:
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        ox, oy, oangle = cx_mean, cy_mean, 0.0
    
    # Compute final side length
    best_side = _min_outer_side_fixed(inner_hexes, ox, oy, oangle)
    
    # Verify all inner hexagons are inside outer
    outer_hex = (ox, oy, best_side, oangle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
    
    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.2
        for _ in range(60):
            mid = (lo + hi) / 2
            test_outer = (ox, oy, mid, oangle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
    
    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    
    return inner, (ox, oy), best_side, oangle
