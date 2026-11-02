"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
import math
import time
import itertools

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


def _convex_hull(points):
    """Compute convex hull of a list of points using Graham scan."""
    if len(points) <= 1:
        return points
    
    # Find the point with lowest y (and lowest x as tiebreaker)
    points = sorted(points, key=lambda p: (p[1], p[0]))
    pivot = points[0]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Sort by polar angle
    rest = []
    for p in points[1:]:
        if p == pivot:
            continue
        rest.append(p)
    
    rest.sort(key=lambda p: (math.atan2(p[1] - pivot[1], p[0] - pivot[0]), 
                              (p[0] - pivot[0])**2 + (p[1] - pivot[1])**2))
    
    if len(rest) == 0:
        return [pivot]
    
    hull = [pivot]
    for p in rest:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    
    # Remove duplicate pivot if present
    if len(hull) > 1 and hull[0] == hull[-1]:
        hull.pop()
    
    return hull


def _min_enclosing_hexagon(points):
    """Find minimum side length of a regular hexagon (any rotation) that encloses all points.
    Returns (side_length, center_x, center_y, angle_degrees)."""
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    # The optimal enclosing hexagon will have at least one side flush with a supporting line
    # of the convex hull. We can try all hull edges and also all rotations from 0 to 60 degrees.
    
    hull = _convex_hull(points)
    n = len(hull)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if n == 1:
        return (0.0, hull[0][0], hull[0][1], 0.0)
    if n == 2:
        # Minimum hexagon enclosing two points
        # The hexagon needs to have the two points inside it
        # For two points, the minimum enclosing regular hexagon...
        # Actually, let's just use the general approach with angle search
        pass
    
    # Try angles from 0 to 60 degrees in small steps
    # The hexagon is symmetric under 60-degree rotation, so we only need 0 to 60
    for angle_deg in range(0, 61, 1):
        a = -math.radians(angle_deg)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        
        # Project all points onto the 6 axes of the hexagon
        # The hexagon centered at (cx, cy) with angle angle_deg contains point (px, py)
        # iff for all 6 directions, the projection is within the apothem
        
        # In the hexagon's local frame, the constraints are:
        # For k=0..5: lx*cos(30+60k) + ly*sin(30+60k) <= apothem
        # where (lx, ly) is the point in the hexagon's local coordinate system
        
        # So: (px-cx)*cos(a)*cos(30+60k) + (px-cx)*sin(a)*sin(30+60k) + 
        #     (py-cy)*sin(a)*cos(30+60k) + (py-cy)*cos(a)*sin(30+60k) <= apothem
        
        # Let's define for each point and each direction:
        # proj_k(px, py) = px * (cos(a)*cos(theta_k) + sin(a)*sin(theta_k)) + 
        #                  py * (sin(a)*cos(theta_k) + cos(a)*sin(theta_k))
        # = px * cos(a - theta_k) + py * sin(a - theta_k) ... wait, let me redo
        
        # Actually, rotating point by -a:
        # lx = (px-cx)*cos(a) - (py-cy)*sin(a)  ... no wait
        # The local coords are obtained by rotating the global coords by -angle_deg
        # lx = (px-cx)*cos(-a_rad) - (py-cy)*sin(-a_rad) = (px-cx)*cos(a_rad) + (py-cy)*sin(a_rad)
        # where a_rad = radians(angle_deg)
        # Hmm, let me be more careful.
        
        # If the hexagon is rotated by angle_deg, then a point (px, py) in global coords
        # has local coords:
        # lx = (px-cx)*cos(angle_rad) + (py-cy)*sin(angle_rad)
        # ly = -(px-cx)*sin(angle_rad) + (py-cy)*cos(angle_rad)
        
        # Wait, let's use the same convention as _point_in_hex:
        # In _point_in_hex, we compute:
        # dx = px - cx, dy = py - cy
        # a = -radians(angle_deg)
        # lx = dx*cos(a) - dy*sin(a)
        # ly = dx*sin(a) + dy*cos(a)
        
        # So with a = -angle_rad:
        # lx = dx*cos(-angle_rad) - dy*sin(-angle_rad) = dx*cos(angle_rad) + dy*sin(angle_rad)
        # ly = dx*sin(-angle_rad) + dy*cos(-angle_rad) = -dx*sin(angle_rad) + dy*cos(angle_rad)
        
        angle_rad = math.radians(angle_deg)
        cos_ar = math.cos(angle_rad)
        sin_ar = math.sin(angle_rad)
        
        # For each point, compute its local coords (without centering)
        # lx_global = px*cos_ar + py*sin_ar
        # ly_global = -px*sin_ar + py*cos_ar
        
        # Then lx = lx_global - (cx*cos_ar + cy*sin_ar)
        # ly = ly_global - (-cx*sin_ar + cy*cos_ar)
        
        # The constraint is: for all k, lx*cos(30+60k) + ly*sin(30+60k) <= apothem
        # = (lx_global - cx*cos_ar - cy*sin_ar)*cos(theta_k) + (ly_global + cx*sin_ar - cy*cos_ar)*sin(theta_k) <= apothem
        # = lx_global*cos(theta_k) + ly_global*sin(theta_k) - cx*(cos_ar*cos(theta_k) - sin_ar*sin(theta_k)) - cy*(sin_ar*cos(theta_k) + cos_ar*sin(theta_k)) <= apothem
        # = lx_global*cos(theta_k) + ly_global*sin(theta_k) - cx*cos(theta_k + angle_rad) - cy*sin(theta_k + angle_rad) <= apothem
        
        # Hmm, this is getting complicated. Let me just use a simpler approach.
        
        # For a given angle, the minimum enclosing hexagon centered at (cx, cy) has:
        # apothem >= max over all points and all 6 directions of:
        #   (point - center) projected onto that direction
        
        # The optimal center for a given angle is the one that minimizes the max projection.
        # This is equivalent to finding the minimum enclosing hexagon, which can be solved
        # by considering that the hexagon is the intersection of 6 half-planes.
        
        # Actually, for a fixed orientation, the minimum enclosing regular hexagon problem
        # is a linear programming problem. But let's use a simpler approach:
        # The center should be at the centroid of the points for a symmetric shape,
        # or we can optimize it.
        
        # Let's try: for each angle, compute the bounding box in the hexagon's local frame,
        # then the apothem needed is max of the 6 support functions.
        
        # For a fixed angle, let's compute the required apothem as a function of center.
        # apothem(cx, cy) = max over all points p and all k of:
        #   [(p.x - cx)*cos_ar + (p.y - cy)*sin_ar]*cos(theta_k) + [-(p.x-cx)*sin_ar + (p.y-cy)*cos_ar]*sin(theta_k)
        
        # = max over p, k of:
        #   p.x*(cos_ar*cos(theta_k) - sin_ar*sin(theta_k)) + p.y*(sin_ar*cos(theta_k) + cos_ar*sin(theta_k)) 
        #   - cx*(cos_ar*cos(theta_k) - sin_ar*sin(theta_k)) - cy*(sin_ar*cos(theta_k) + cos_ar*sin(theta_k))
        
        # = max over p, k of:
        #   p.x*cos(theta_k - angle_rad) + p.y*sin(theta_k - angle_rad) 
        #   - cx*cos(theta_k - angle_rad) - cy*sin(theta_k - angle_rad)
        
        # Wait, that's not right either. Let me just compute it numerically.
        
        # For each point, compute local coords (with center at origin for now)
        lx_vals = []
        ly_vals = []
        for px, py in points:
            lx = px * cos_ar + py * sin_ar
            ly = -px * sin_ar + py * cos_ar
            lx_vals.append(lx)
            ly_vals.append(ly)
        
        # For center (cx, cy), the local coords become (lx - cx*cos_ar - cy*sin_ar, ly + cx*sin_ar - cy*cos_ar)
        # Let cx_l = cx*cos_ar + cy*sin_ar, cy_l = -cx*sin_ar + cy*cos_ar
        # Then local coords are (lx - cx_l, ly - cy_l)
        
        # The 6 support values for a given center (cx_l, cy_l) are:
        # max_p [(lx_p - cx_l)*cos(theta_k) + (ly_p - cy_l)*sin(theta_k)]
        # = max_p [lx_p*cos(theta_k) + ly_p*sin(theta_k)] - cx_l*cos(theta_k) - cy_l*sin(theta_k)
        
        # So we need: for all k, max_p [lx_p*cos(theta_k) + ly_p*sin(theta_k)] - cx_l*cos(theta_k) - cy_l*sin(theta_k) <= apothem
        
        # To minimize apothem, we want to choose cx_l, cy_l to minimize the max over k of:
        #   max_p [lx_p*cos(theta_k) + ly_p*sin(theta_k)] - cx_l*cos(theta_k) - cy_l*sin(theta_k)
        
        # This is a 2D linear programming problem. The optimal (cx_l, cy_l) is at the intersection
        # of two of the lines defining the constraints.
        
        # For simplicity, let's try the centroid and a few other centers.
        cx_l_centroid = sum(lx_vals) / len(lx_vals)
        cy_l_centroid = sum(ly_vals) / len(ly_vals)
        
        # Also try the center of the bounding box
        lx_min = min(lx_vals)
        lx_max = max(lx_vals)
        ly_min = min(ly_vals)
        ly_max = max(ly_vals)
        cx_l_bbox = (lx_min + lx_max) / 2
        cy_l_bbox = (ly_min + ly_max) / 2
        
        candidates_center = [
            (cx_l_centroid, cy_l_centroid),
            (cx_l_bbox, cy_l_bbox),
        ]
        
        # Also try centers that are midpoints of pairs of extreme points
        # For each pair of hull vertices, try the midpoint
        if len(hull) >= 2:
            for i in range(len(hull)):
                for j in range(i + 1, len(hull)):
                    hx1, hy1 = hull[i]
                    hx2, hy2 = hull[j]
                    hx_l1 = hx1 * cos_ar + hy1 * sin_ar
                    hy_l1 = -hx1 * sin_ar + hy1 * cos_ar
                    hx_l2 = hx2 * cos_ar + hy2 * sin_ar
                    hy_l2 = -hx2 * sin_ar + hy2 * cos_ar
                    candidates_center.append(((hx_l1 + hx_l2) / 2, (hy_l1 + hy_l2) / 2))
        
        for cx_l, cy_l in candidates_center:
            max_proj = 0.0
            for k in range(6):
                theta_k = math.radians(30 + 60 * k)
                cos_tk = math.cos(theta_k)
                sin_tk = math.sin(theta_k)
                
                # max over points of (lx_p - cx_l)*cos_tk + (ly_p - cy_l)*sin_tk
                max_p = -float('inf')
                for lx_p, ly_p in zip(lx_vals, ly_vals):
                    val = (lx_p - cx_l) * cos_tk + (ly_p - cy_l) * sin_tk
                    if val > max_p:
                        max_p = val
                if max_p > max_proj:
                    max_proj = max_p
            
            side = max_proj * 2.0 / math.sqrt(3)
            if side < best_side:
                best_side = side
                # Convert center back to global coords
                # cx_l = cx*cos_ar + cy*sin_ar
                # cy_l = -cx*sin_ar + cy*cos_ar
                # Solve for cx, cy:
                # [cos_ar  sin_ar] [cx]   [cx_l]
                # [-sin_ar cos_ar] [cy] = [cy_l]
                # This is a rotation matrix, so the inverse is its transpose:
                # cx = cx_l*cos_ar - cy_l*sin_ar
                # cy = cx_l*sin_ar + cy_l*cos_ar
                best_center = (cx_l * cos_ar - cy_l * sin_ar, cx_l * sin_ar + cy_l * cos_ar)
                best_angle = angle_deg
    
    # Also try finer angle search around the best angle found
    if best_side < float('inf'):
        for delta_angle in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            angle_deg = (best_angle + delta_angle) % 60
            if angle_deg < 0:
                angle_deg += 60
            
            angle_rad = math.radians(angle_deg)
            cos_ar = math.cos(angle_rad)
            sin_ar = math.sin(angle_rad)
            
            lx_vals = []
            ly_vals = []
            for px, py in points:
                lx = px * cos_ar + py * sin_ar
                ly = -px * sin_ar + py * cos_ar
                lx_vals.append(lx)
                ly_vals.append(ly)
            
            cx_l_centroid = sum(lx_vals) / len(lx_vals)
            cy_l_centroid = sum(ly_vals) / len(ly_vals)
            
            candidates_center = [(cx_l_centroid, cy_l_centroid)]
            
            if len(hull) >= 2:
                for i in range(len(hull)):
                    for j in range(i + 1, len(hull)):
                        hx1, hy1 = hull[i]
                        hx2, hy2 = hull[j]
                        hx_l1 = hx1 * cos_ar + hy1 * sin_ar
                        hy_l1 = -hx1 * sin_ar + hy1 * cos_ar
                        hx_l2 = hx2 * cos_ar + hy2 * sin_ar
                        hy_l2 = -hx2 * sin_ar + hy2 * cos_ar
                        candidates_center.append(((hx_l1 + hx_l2) / 2, (hy_l1 + hy_l2) / 2))
            
            for cx_l, cy_l in candidates_center:
                max_proj = 0.0
                for k in range(6):
                    theta_k = math.radians(30 + 60 * k)
                    cos_tk = math.cos(theta_k)
                    sin_tk = math.sin(theta_k)
                    
                    max_p = -float('inf')
                    for lx_p, ly_p in zip(lx_vals, ly_vals):
                        val = (lx_p - cx_l) * cos_tk + (ly_p - cy_l) * sin_tk
                        if val > max_p:
                            max_p = val
                    if max_p > max_proj:
                        max_proj = max_p
                
                side = max_proj * 2.0 / math.sqrt(3)
                if side < best_side:
                    best_side = side
                    best_center = (cx_l * cos_ar - cy_l * sin_ar, cx_l * sin_ar + cy_l * cos_ar)
                    best_angle = angle_deg
    
    return (best_side, best_center[0], best_center[1], best_angle)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
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

    # Try all combinations of 4 from ring2_candidates
    for combo in itertools.combinations(range(len(ring2_candidates)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2_candidates[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Collect all vertices of all hexagons
        all_points = []
        for x, y in full_xy:
            verts = _hex_vertices(x, y, 1.0, 0.0)
            all_points.extend(verts)
        
        # Find minimum enclosing hexagon
        side, ocx, ocy, oang = _min_enclosing_hexagon(all_points)
        
        if side < best_side:
            best_side = side
            best_xy = full_xy
            best_center = (ocx, ocy)
            best_angle = oang

    if best_xy is None:
        # Fallback
        best_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        all_points = []
        for x, y in best_xy:
            verts = _hex_vertices(x, y, 1.0, 0.0)
            all_points.extend(verts)
        best_side, best_center[0], best_center[1], best_angle = _min_enclosing_hexagon(all_points)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        all_points = []
        for x, y, s, a in inner_hexes:
            verts = _hex_vertices(x, y, s, a)
            all_points.extend(verts)
        best_side, best_center[0], best_center[1], best_angle = _min_enclosing_hexagon(all_points)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.005

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
