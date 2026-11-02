"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses a hexagonal lattice with correct spacing to ensure disjointness,
then optimizes the outer hexagon center and rotation.
"""
import math
import time
import itertools

N = 11
SQRT3 = math.sqrt(3.0)


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
    apothem = side * SQRT3 / 2.0
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
    return max_val * 2.0 / SQRT3


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # Hexagonal lattice: for unit hexagons, adjacent centers are at distance SQRT3
    # Flat-top orientation: x = SQRT3 * (col + row/2), y = 1.5 * row
    # This gives adjacent distance of sqrt(3) between centers
    
    def lattice_to_xy(col, row):
        x = SQRT3 * (col + row * 0.5)
        y = 1.5 * row
        return (x, y)

    # Generate positions: center + ring1 (6 neighbors) + select 4 from ring2
    center = (0, 0)
    # Ring 1: 6 neighbors in axial coordinates
    ring1_axial = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Ring 2: 12 positions at distance 2
    ring2_axial = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)
    ]

    # Convert axial to (col, row) for lattice_to_xy
    # Axial (q, r) -> col = q, row = r (this works for the standard mapping)
    
    base_positions = [center] + ring1_axial
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2
    n_r2 = len(ring2_axial)
    for combo in itertools.combinations(range(n_r2), 4):
        if time.time() > deadline:
            break
        
        selected_r2 = [ring2_axial[i] for i in combo]
        full_axial = base_positions + selected_r2
        
        # Convert to Cartesian
        full_xy = [lattice_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Try different outer hexagon configurations
        candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 30.0),
            (cx_mean, cy_mean, 15.0),
            (cx_mean, cy_mean, 45.0),
        ]
        
        for ocx, ocy, oang in candidates:
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
            side = _min_outer_side(inner_hexes, ocx, ocy, oang)
            if side < best_side:
                best_side = side
                best_xy = full_xy
                best_center = (ocx, ocy)
                best_angle = oang

    if best_xy is None:
        # Fallback
        best_xy = [lattice_to_xy(q, r) for q, r in base_positions]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_xy], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Scale up positions slightly to ensure separation
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
        best_center = (cx_mean, cy_mean)
        best_angle = 0.0

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

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
