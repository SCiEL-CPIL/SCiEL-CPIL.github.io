"""Optimized hexagon packing: 11 unit hexagons with LP-based outer optimization
and Nelder-Mead local search for inner positions and rotations."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

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
    Uses 2D linear programming.
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
    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    # Linear programming: minimize A
    # Variables: ox, oy, A (3 variables)
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
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, 
                        bounds=[(None, None), (None, None), (0, None)], 
                        method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    # Fallback: use centroid
    cx_mean = np.mean(pts[:, 0])
    cy_mean = np.mean(pts[:, 1])
    max_proj = 0.0
    for k in range(6):
        ux, uy = u_vectors[k]
        for px, py in pts:
            proj = (px - cx_mean) * ux + (py - cy_mean) * uy
            if abs(proj) > max_proj:
                max_proj = abs(proj)
    side = max_proj * 2.0 / math.sqrt(3)
    return side, cx_mean, cy_mean


def _objective(params, hex_positions):
    """
    Objective function for Nelder-Mead optimization.
    params: [x0, y0, x1, y1, ..., x10, y10, a0, a1, ..., a10]
    where (xi, yi) are inner hexagon centers and ai are their angles (degrees)
    Returns the outer hexagon side length (to minimize)
    """
    n = len(hex_positions)
    centers = [(params[2*i], params[2*i+1]) for i in range(n)]
    angles = [params[n*2 + i] for i in range(n)]
    
    # Build inner hexagons
    inner_hexes = [(cx, cy, 1.0, ang) for (cx, cy), ang in zip(centers, angles)]
    
    # Check disjointness - if overlapping, return large penalty
    if not _all_disjoint(inner_hexes):
        # Compute minimum violation as penalty
        min_dist = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
        return 10.0 + max(0, 2.0 - min_dist)  # Penalty for overlap
    
    # Compute minimum outer side
    side, _, _ = _min_outer_side_analytical(inner_hexes, 0.0)
    return side


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Define 11 positions in axial coordinates
    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Phase 1: Try lattice-based configurations
    for combo in itertools.combinations(range(len(ring2_candidates)), 4):
        if time.time() > deadline - 60:
            break
            
        selected_r2 = [ring2_candidates[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Try a few outer rotations
        for angle_deg in [0.0, 15.0, 30.0]:
            angle_rad = math.radians(angle_deg)
            side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, angle_rad)
            if side < best_side:
                best_side = side
                best_xy = full_xy
                best_angles = [0.0] * 11
                best_center = (ox + cx_mean, oy + cy_mean)
                best_angle = angle_deg

    # Phase 2: Local search with Nelder-Mead on the best lattice configuration
    if time.time() < deadline - 30 and best_xy is not None:
        # Use the best lattice configuration as starting point
        cx_mean = sum(x for x, y in best_xy) / 11.0
        cy_mean = sum(y for x, y in best_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        
        # Initial parameters: centers and angles
        x0 = []
        for cx, cy in centered_xy:
            x0.extend([cx, cy])
        x0.extend([0.0] * 11)  # all angles start at 0
        
        # Try Nelder-Mead optimization
        try:
            result = minimize(
                _objective, x0, args=(best_xy,),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if result.fun < best_side:
                # Extract optimized positions
                n = 11
                opt_centers = [(result.x[2*i], result.x[2*i+1]) for i in range(n)]
                opt_angles = [result.x[n*2 + i] for i in range(n)]
                
                # Convert back to original frame
                opt_xy = [(cx + cx_mean, cy + cy_mean) for cx, cy in opt_centers]
                
                # Verify disjointness
                inner_hexes_opt = [(cx, cy, 1.0, ang) for (cx, cy), ang in zip(opt_xy, opt_angles)]
                if _all_disjoint(inner_hexes_opt):
                    best_side = result.fun
                    best_xy = opt_xy
                    best_angles = opt_angles
        except Exception:
            pass

    # Phase 3: Fine-tune outer hexagon for the best configuration
    if best_xy is not None:
        cx_mean = sum(x for x, y in best_xy) / 11.0
        cy_mean = sum(y for x, y in best_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, ang) for (x, y), ang in zip(centered_xy, best_angles)]
        
        # Sweep outer rotation more finely
        for angle_deg in np.arange(0, 30, 0.5):
            if time.time() > deadline - 5:
                break
            angle_rad = math.radians(angle_deg)
            side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, angle_rad)
            if side < best_side:
                best_side = side
                best_center = (ox + cx_mean, oy + cy_mean)
                best_angle = angle_deg

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, ang) for (x, y), ang in zip(best_xy, best_angles)]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Fallback to lattice-only
        pos_axial = [(0, 0)]
        pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
        best_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        best_angles = [0.0] * 11
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y in best_xy) / 11.0
        cy_mean = sum(y for x, y in best_xy) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
