"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def cross(o, a, b):
    """Cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
        min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    
    return False

def polygons_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons have overlapping interiors."""
    n1 = len(verts1)
    n2 = len(verts2)
    
    # Check if any vertex of poly1 is inside poly2
    for v in verts1:
        inside = True
        for i in range(n2):
            if cross(verts2[i], verts2[(i+1)%n2], v) < -tol:
                inside = False
                break
        if inside:
            return True
    
    # Check if any vertex of poly2 is inside poly1
    for v in verts2:
        inside = True
        for i in range(n1):
            if cross(verts1[i], verts1[(i+1)%n1], v) < -tol:
                inside = False
                break
        if inside:
            return True
    
    # Check if any edges intersect
    for i in range(n1):
        for j in range(n2):
            if segments_intersect(verts1[i], verts1[(i+1)%n1], verts2[j], verts2[(j+1)%n2]):
                return True
    
    return False

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside or on the boundary of the hexagon."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    inradius = side * np.sqrt(3) / 2
    for i in range(6):
        normal_angle = np.radians(30 + i * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True

def is_valid(inner, center, side, angle, tol=1e-6):
    """Check if the packing is valid."""
    cx, cy = center
    outer_verts = hexagon_vertices(cx, cy, side, angle)
    
    # Check all inner hexagons are inside the outer hexagon
    for x, y, a in inner:
        verts = hexagon_vertices(x, y, 1.0, a)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], cx, cy, side, angle):
                return False
    
    # Check no two inner hexagons overlap
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_overlap(v1, v2):
                return False
    
    return True

def outer_side_length(inner, center, angle):
    """Compute minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = center
    max_side = 0.0
    for x, y, a in inner:
        verts = hexagon_vertices(x, y, 1.0, a)
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(angle)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for i in range(6):
                normal_angle = np.radians(30 + i * 60)
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                val = abs(lx * nx + ly * ny)
                required_side = 2.0 * val / np.sqrt(3)
                if required_side > max_side:
                    max_side = required_side
    
    return max_side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    d = np.sqrt(3)  # center-to-center distance for adjacent unit hexes
    
    # Honeycomb initial guess
    honeycomb = []
    honeycomb.append([0.0, 0.0, 0.0])
    for i in range(6):
        a = i * np.pi / 3
        honeycomb.append([d * np.cos(a), d * np.sin(a), 0.0])
    for i in range(5):
        a = (i + 0.5) * np.pi / 3
        r2 = d * np.sqrt(3)
        honeycomb.append([r2 * np.cos(a), r2 * np.sin(a), 0.0])
    
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = honeycomb[i][0]
        params[3*i+1] = honeycomb[i][1]
        params[3*i+2] = honeycomb[i][2]
    
    center = (0.0, 0.0)
    angle_outer = 0.0
    
    best_side = outer_side_length(
        [[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)], center, angle_outer
    )
    best_params = params.copy()
    
    # Try optimization
    try:
        result = minimize(
            lambda p: outer_side_length([[p[3*i], p[3*i+1], p[3*i+2]] for i in range(N)], center, angle_outer),
            params,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-7}
        )
        
        if result.fun < best_side:
            best_side = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    
    # Ensure validity
    for _ in range(50):
        if is_valid(inner, center, best_side, angle_outer):
            break
        best_side *= 1.02
    
    return inner, center, best_side, angle_outer

# EVOLVE-BLOCK-END
