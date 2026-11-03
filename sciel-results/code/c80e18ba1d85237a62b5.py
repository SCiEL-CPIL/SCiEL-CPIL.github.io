"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern."""
import math
import time
import itertools
import numpy as np

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
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
    return max_val * 2.0 / math.sqrt(3)


def _outer_side_smooth(hexes, ox, oy, oangle=0.0, beta=100.0):
    """Smooth surrogate for _min_outer_side using log-sum-exp over all vertex projections.
    As beta -> inf, this converges to the exact max. Used for gradient-based optimization."""
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    # Pre-compute the 6 normal directions
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Collect all projections
    projs = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for nx, ny in normals:
                projs.append(lx * nx + ly * ny)
    
    if not projs:
        return 0.0
    
    projs = np.array(projs)
    # Log-sum-exp for numerical stability
    m = np.max(projs)
    lse = m + np.log(np.sum(np.exp(beta * (projs - m)))) / beta
    return lse * 2.0 / math.sqrt(3)


def _optimize_outer_smooth(hexes, ox0, oy0, a0, deadline, max_iter=500):
    """Use L-BFGS-B on the smooth log-sum-exp surrogate to find optimal outer hexagon
    center and rotation. Returns (cx, cy, angle_deg, exact_side)."""
    from scipy.optimize import minimize
    
    def smooth_obj(params):
        cx, cy, ang = params
        return _outer_side_smooth(hexes, cx, cy, ang, beta=100.0)
    
    x0 = np.array([ox0, oy0, math.radians(a0)])
    result = minimize(smooth_obj, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-15})
    
    if result.success or result.fun < smooth_obj(x0):
        cx, cy, ang_rad = result.x
        ang_deg = math.degrees(ang_rad)
        # Compute exact side at this location
        exact_side = _min_outer_side(hexes, cx, cy, ang_deg)
        return cx, cy, ang_deg, exact_side
    return ox0, oy0, a0, _min_outer_side(hexes, ox0, oy0, a0)


def _high_precision_polish(hexes, ox, oy, oangle, deadline):
    """Iterative high-precision polish: alternate between smooth optimization and
    exact evaluation, with progressively tighter tolerances."""
    cur_cx, cur_cy, cur_ang, cur_side = ox, oy, oangle, _min_outer_side(hexes, ox, oy, oangle)
    
    # Multiple refinement rounds with increasing beta
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() > deadline - 5.0:
            break
        # Use smooth surrogate to find a better point
        new_cx, new_cy, new_ang, new_side = _optimize_outer_smooth(
            hexes, cur_cx, cur_cy, cur_ang, deadline, max_iter=200)
        if new_side < cur_side - 1e-14:
            cur_cx, cur_cy, cur_ang, cur_side = new_cx, new_cy, new_ang, new_side
    
    # Final exact local search with very small steps
    for step in [0.001, 0.0001, 0.00001, 0.000001]:
        if time.time() > deadline - 2.0:
            break
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    for da in [-step*10, 0, step*10]:
                        ncx, ncy, nang = cur_cx+dx, cur_cy+dy, cur_ang+da
                        s = _min_outer_side(hexes, ncx, ncy, nang)
                        if s < cur_side - 1e-15:
                            cur_cx, cur_cy, cur_ang, cur_side = ncx, ncy, nang, s
                            improved = True
    
    return cur_cx, cur_cy, cur_ang, cur_side


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse grid to find promising configurations
    for combo in all_combos:
        if time.time() > deadline - 40.0:  # Leave 40s for optimization phases
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations
        for oang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0
    
    # Phase 2: Smooth gradient-based optimization using log-sum-exp surrogate
    if best_positions is not None and time.time() < deadline - 30.0:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_opt, cy_opt, ang_opt, side_opt = _optimize_outer_smooth(
            inner_hexes, best_center[0], best_center[1], best_angle, deadline, max_iter=300)
        if side_opt < best_side:
            best_side = side_opt
            best_center = (cx_opt, cy_opt)
            best_angle = ang_opt
    
    # Phase 3: High-precision polish with alternating smooth/exact refinement
    if best_positions is not None and time.time() < deadline - 10.0:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_final, cy_final, ang_final, side_final = _high_precision_polish(
            inner_hexes, best_center[0], best_center[1], best_angle, deadline)
        if side_final < best_side:
            best_side = side_final
            best_center = (cx_final, cy_final)
            best_angle = ang_final

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness - with proper axial spacing, this should always pass
    # In a hexagonal lattice with flat-top hexagons of side 1, adjacent hexagons
    # share an edge and have disjoint interiors.
    
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        # This shouldn't happen with correct axial mapping
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    # Verify all inner hexagons are inside outer with tight tolerance
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
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
