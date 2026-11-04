import math
import time
import random
import numpy as np
from scipy.optimize import differential_evolution

N = 12
TIME_LIMIT = 135.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + tol or maxB <= minA + tol:
            return False
    return True

def _hex_normals(angle_deg):
    """
    Pre-computed edge normals for a unit hexagon with given rotation.
    The edges of a regular hexagon (vertices at 0, 60, ...) have normals at 30, 90, 150, 210, 270, 330.
    """
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    normals = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Rotate normal
        rnx = ca * nx - sa * ny
        rny = sa * nx + ca * ny
        normals.append((rnx, rny))
    return normals

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    normals = [_hex_normals(p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            axes = normals[i] + normals[j]
            sep = False
            for axis in axes:
                minA, maxA = _project_polygon(polys[i], axis)
                minB, maxB = _project_polygon(polys[j], axis)
                if maxA <= minB + tol or maxB <= minA + tol:
                    sep = True
                    break
            if not sep:
                return False
    return True

def _sdf_distance(p1x, p1y, a1, p2x, p2y, a2):
    """
    Compute the signed distance between two unit hexagons.
    Positive means separated, negative means overlapping.
    """
    dx = p2x - p1x
    dy = p2y - p1y
    dist = math.hypot(dx, dy)
    
    if dist < 1e-10:
        return 0.0
    
    def hex_radius(angle_deg, direction_deg):
        a_rad = math.radians(angle_deg)
        d_rad = math.radians(direction_deg)
        dx_dir = math.cos(d_rad)
        dy_dir = math.sin(d_rad)
        
        max_val = 0.0
        for k in range(6):
            theta = math.radians(60 * k)
            vx = math.cos(theta)
            vy = math.sin(theta)
            ca = math.cos(a_rad)
            sa = math.sin(a_rad)
            rvx = ca * vx - sa * vy
            rvy = sa * vx + ca * vy
            dot = rvx * dx_dir + rvy * dy_dir
            if dot > max_val:
                max_val = dot
        return max_val
    
    dir_deg = math.degrees(math.atan2(dy, dx))
    r1 = hex_radius(a1, dir_deg)
    r2 = hex_radius(a2, dir_deg + 180)
    
    return dist - r1 - r2

def _make_hexagonal_arrangement():
    """Create a hexagonal lattice arrangement of 12 unit hexagons."""
    inner = []
    d = math.sqrt(3)  # distance between adjacent hex centers in hex lattice
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons at distance d, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        theta = math.radians(60 * k)
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        inner.append([x, y, 0.0])
    
    # Second ring: 5 more hexagons. Place them at the "gap" positions.
    for k in range(5):
        theta = math.radians(60 * k + 30)
        x = 2 * d * math.cos(theta)
        y = 2 * d * math.sin(theta)
        inner.append([x, y, 0.0])
    
    return inner[:12]

def _make_flower():
    """Flower pattern: 1 center + 6 inner ring + 5 outer."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    R1 = math.sqrt(3)
    for i in range(6):
        theta = 2 * math.pi * i / 6
        x = R1 * math.cos(theta)
        y = R1 * math.sin(theta)
        pts.append([x, y, 0.0])
    R2 = 2 * math.sqrt(3)
    for i in range(5):
        theta = 2 * math.pi * (i + 0.5) / 5
        x = R2 * math.cos(theta)
        y = R2 * math.sin(theta)
        pts.append([x, y, 0.0])
    return pts

def _make_dense_cluster():
    """Tight cluster arrangement."""
    pts = []
    pts.append([0.0, 0.0, 0.0])
    for i in range(6):
        theta = 2 * math.pi * i / 6
        x = math.sqrt(3) * math.cos(theta)
        y = math.sqrt(3) * math.sin(theta)
        pts.append([x, y, 0.0])
    for i in range(5):
        theta = 2 * math.pi * (i + 0.5) / 5
        x = 2 * math.sqrt(3) * math.cos(theta)
        y = 2 * math.sqrt(3) * math.sin(theta)
        pts.append([x, y, 0.0])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with hexagonal arrangement
    inner = _make_hexagonal_arrangement()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial outer side
    outer_side = _outer_hex_side(inner, outer_center, outer_angle)
    
    best_inner = [list(x) for x in inner]
    best_center = outer_center
    best_side = outer_side
    best_angle = outer_angle
    
    # Try several initial arrangements
    for init_func in [_make_hexagonal_arrangement, _make_flower, _make_dense_cluster]:
        if time.time() > deadline:
            break
        init_inner = init_func()
        if not _is_valid(init_inner):
            continue
        init_side = _outer_hex_side(init_inner, (0.0, 0.0), 0.0)
        if init_side < best_side:
            best_side = init_side
            best_inner = [list(x) for x in init_inner]
            inner = [list(x) for x in init_inner]
            outer_side = init_side
    
    # Simulated annealing on inner positions and angles
    temp = 2.0
    cooling = 0.999
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try a random perturbation
        idx = random.randint(0, N - 1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 5.0)
        
        new_inner = [list(x) for x in inner]
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        # Check disjointness (expensive, but necessary)
        if not _is_valid(new_inner):
            temp *= cooling
            continue
        
        # Compute new outer side with current center and angle
        new_side = _outer_hex_side(new_inner, outer_center, outer_angle)
        
        delta = new_side - outer_side
        
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
            inner = new_inner
            outer_side = new_side
            if outer_side < best_side:
                best_side = outer_side
                best_inner = [list(x) for x in inner]
        
        temp *= cooling
    
    # Try a few different outer angles and centers
    for angle_try in [0, 30, 60]:
        if time.time() > deadline:
            break
        outer_angle_try = float(angle_try)
        # Try to optimize center
        for cx_try in [-1.0, 0.0, 1.0]:
            for cy_try in [-1.0, 0.0, 1.0]:
                if time.time() > deadline:
                    break
                oc = (cx_try, cy_try)
                side_try = _outer_hex_side(best_inner, oc, outer_angle_try)
                if side_try < best_side:
                    # Verify validity
                    if _is_valid(best_inner):
                        best_side = side_try
                        best_center = oc
                        best_angle = outer_angle_try
    
    return best_inner, best_center, best_side, best_angle
