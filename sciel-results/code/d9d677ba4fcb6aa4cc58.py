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

def _min_outer_side_optimized(vertices, ox, oy, oangle):
    """
    Calculate the minimal side length of an outer hexagon centered at (ox, oy) 
    with rotation oangle that contains all given vertices.
    vertices: list of (x, y)
    """
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    # Precompute normals for the outer hexagon
    # Normals are at 30, 90, 150, 210, 270, 330 degrees relative to the hexagon's frame
    # In the global frame, they are rotated by oangle
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k + oangle)
        normals.append((math.cos(theta), math.sin(theta)))
        
    for vx, vy in vertices:
        dx = vx - ox
        dy = vy - oy
        for nx, ny in normals:
            proj = dx * nx + dy * ny
            if proj > max_val:
                max_val = proj
                
    return max_val * 2.0 / math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    sqrt3 = math.sqrt(3)

    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_config = None

    # Phase 1: Coarse search to find promising combinations
    candidates = []
    for combo in itertools.combinations(range(len(ring2_candidates)), 4):
        if time.time() > deadline - 20: # Reserve time for refinement
            break
            
        selected_r2 = [ring2_candidates[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Get all vertices of the 11 hexagons
        all_verts = []
        for x, y in full_xy:
            all_verts.extend(_hex_vertices(x, y, 1.0, 0.0))
            
        # Centroid of vertices
        cx_mean = sum(v[0] for v in all_verts) / len(all_verts)
        cy_mean = sum(v[1] for v in all_verts) / len(all_verts)
        
        # Try a few coarse angles
        for oang in [0, 10, 20, 30, 40, 50]:
            side = _min_outer_side_optimized(all_verts, cx_mean, cy_mean, oang)
            candidates.append((side, full_xy, oang, (cx_mean, cy_mean)))
            
    if not candidates:
        # Fallback
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        all_verts = []
        for x, y in full_xy:
            all_verts.extend(_hex_vertices(x, y, 1.0, 0.0))
        cx_mean = sum(v[0] for v in all_verts) / len(all_verts)
        cy_mean = sum(v[1] for v in all_verts) / len(all_verts)
        side = _min_outer_side_optimized(all_verts, cx_mean, cy_mean, 0)
        return [[x, y, 0.0] for x, y in full_xy], (cx_mean, cy_mean), side, 0.0

    # Sort by side length and take top 10
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:10]
    
    # Phase 2: Fine search on top candidates
    for side_coarse, full_xy, oang_coarse, center_coarse in top_candidates:
        if time.time() > deadline - 5:
            break
            
        all_verts = []
        for x, y in full_xy:
            all_verts.extend(_hex_vertices(x, y, 1.0, 0.0))
            
        # Refine angle
        for oang in np.arange(oang_coarse - 5, oang_coarse + 5, 0.5):
            if oang < 0: oang += 60
            if oang >= 60: oang -= 60
            
            # Refine center: Grid search around the coarse center
            # The optimal center is likely close to the centroid
            # We can do a few steps of gradient descent or just a small grid
            best_local_side = float('inf')
            best_local_center = center_coarse
            
            # Simple grid search for center
            # Step size 0.1, range 1.0
            for dx in np.arange(-1.0, 1.0, 0.2):
                for dy in np.arange(-1.0, 1.0, 0.2):
                    ox = center_coarse[0] + dx
                    oy = center_coarse[1] + dy
                    side = _min_outer_side_optimized(all_verts, ox, oy, oang)
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (ox, oy)
                        
            if best_local_side < best_side:
                best_side = best_local_side
                best_config = (full_xy, best_local_center, oang)

    if best_config is None:
        # Use the best from coarse if refinement failed
        side_coarse, full_xy, oang_coarse, center_coarse = candidates[0]
        best_side = side_coarse
        best_config = (full_xy, center_coarse, oang_coarse)

    full_xy, best_center, best_angle = best_config
    
    # Final verification and adjustment
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
    
    # Ensure containment with a small safety margin
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break
            
    if not valid:
        best_side *= 1.001
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)
        
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]
    return inner, (best_center[0], best_center[1]), best_side, best_angle
