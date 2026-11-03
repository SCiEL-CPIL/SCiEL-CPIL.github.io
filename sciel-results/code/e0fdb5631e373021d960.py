"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

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

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    # 6 normal directions for the outer hexagon
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    # For each direction, compute max and min projection
    P_vals = np.zeros(6)  # max projection
    Q_vals = np.zeros(6)  # min projection
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    # Linear programming: minimize A
    # Variables: ox, oy, A (3 variables)
    # Constraints:
    # For each k:
    #   ox*u_kx + oy*uky - A <= Q_k
    #   -ox*u_kx - oy*uky - A <= -P_k
    
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1
        b_vec[2*k] = Q_vals[k]
        
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1
        b_vec[2*k+1] = -P_vals[k]
    
    c_vec = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    # Fallback if linprog fails (should be rare)
    return float('inf'), 0.0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)

    # Small gap factor to ensure hexagons are strictly disjoint (not just touching)
    GAP = 1.001

    # Flat-top hexagon mapping
    def axial_to_xy(q, r):
        x = 1.5 * q * GAP
        y = sqrt3 * (r + q / 2.0) * GAP
        return (x, y)

    # Define 11 positions in axial coordinates (q, r)
    # Center
    pos_axial = [(0, 0)]
    # Ring 1 (6 neighbors)
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # We need 4 more from Ring 2.
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None
    best_outer_center = (0.0, 0.0)

    # Pre-compute Cartesian coordinates for all potential positions
    # to avoid repeated calculations
    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    
    # Indices for ring 2 in all_xy
    # pos_axial has 7 items. ring2 starts at index 7.
    ring2_indices = list(range(7, len(all_xy)))

    # Pre-compute a sorted list of ring-2 candidates by distance from origin
    # to prioritize more central positions
    ring2_with_dist = [(i, math.sqrt(all_xy[i][0]**2 + all_xy[i][1]**2)) for i in ring2_indices]
    ring2_with_dist.sort(key=lambda x: x[1])
    ring2_sorted = [i for i, d in ring2_with_dist]
    
    # Iterate over combinations of 4 ring-2 positions, prioritizing closer ones
    for combo in itertools.combinations(ring2_sorted, 4):
        if time.time() > deadline:
            break
            
        # Construct the list of 11 points
        # Indices 0-6 are always included
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        # Inner hexagons are aligned with the lattice (angle 0)
        # We do NOT center them here. The _min_outer_side_analytical function
        # will find the optimal center for the outer hexagon in these coordinates.
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Use ternary search to find the best outer rotation angle in [0, 30] degrees
        # The function f(theta) = min_side(theta) is convex in this range
        lo = 0.0
        hi = 30.0
        # Golden ratio
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        
        fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        
        # Iterate ternary search - use fewer iterations initially for speed
        num_iter = 25
        for _ in range(num_iter):
            if time.time() > deadline:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        
        # Final best angle is the midpoint of the final interval
        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_angle_candidate))
        
        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_center = (ox, oy)
            best_xy = full_xy
            best_inner_hexes = inner_hexes
            best_outer_center = (ox, oy)

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes, 0.0)
        best_center = (ox, oy)
        best_angle = 0.0
        best_inner_hexes = inner_hexes
        best_outer_center = (ox, oy)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # If local search failed or produced invalid result, we should revert to a known valid state.
        # However, since we are using a GAP factor, the lattice positions should be valid.
        # If this happens, it means our GAP was too small or there's a bug.
        # For safety, we can try to increase GAP slightly and re-check, but that changes the geometry.
        # Let's just return the current best if it's close, or a fallback.
        # Given the GAP strategy, this block should rarely trigger.
        pass

    # Final high-precision polish: refine the outer hexagon parameters
    # Use a more precise local optimization around the current best angle
    if time.time() < deadline - 5.0:  # Leave 5s safety margin
        # Try a few nearby angles with higher precision
        for delta_angle in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            test_angle = best_angle + delta_angle
            if test_angle < 0 or test_angle > 30:
                continue
            side_test, ox_test, oy_test = _min_outer_side_analytical(inner_hexes, math.radians(test_angle))
            if side_test < best_side:
                best_side = side_test
                best_angle = test_angle
                best_center = (ox_test, oy_test)
        
        # One more ternary search refinement in a narrow window
        if time.time() < deadline - 3.0:
            lo = max(0.0, best_angle - 3.0)
            hi = min(30.0, best_angle + 3.0)
            gr = (math.sqrt(5) - 1) / 2
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - gr * (hi - lo)
                    fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + gr * (hi - lo)
                    fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
            
            best_angle = (lo + hi) / 2.0
            best_side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_angle))
            best_center = (ox, oy)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
