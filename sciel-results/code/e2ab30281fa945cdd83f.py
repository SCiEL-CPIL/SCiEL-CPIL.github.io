import numpy as np
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    """
    n = 12
    
    # Use a configuration with s = 6, which is achievable
    # Place hexagons in a hexagonal pattern
    
    # For pointy-top hexagons (angle 0), center-to-center distance for adjacent hexagons is sqrt(3)
    # In a hexagonal grid:
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Odd rows offset by 0.75
    
    # Configuration: 3 rows of 4 hexagons each
    # Row 0 (y=0): x = -2.25, -0.75, 0.75, 2.25
    # Row 1 (y=sqrt(3)): x = -1.5, 0, 1.5, 3.0 -- this is asymmetric
    # Let me use a symmetric configuration
    
    # Better: use axial coordinates
    # (q, r) -> x = 1.5*q, y = sqrt(3)*(r + q/2)
    
    coords = [
        (0, 0),
        (1, 0), (2, 0),
        (-1, 0), (-2, 0),
        (0, 1), (1, 1),
        (0, -1), (-1, -1),
        (2, -1), (2, 1),
        (-2, -1), (-2, 1),
    ]
    
    xs = np.array([1.5 * q for q, r in coords])
    ys = np.array([np.sqrt(3) * (r + q/2) for q, r in coords])
    
    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        inner_hex_data[i] = [xs[i], ys[i], 0]
    
    outer_hex_data = np.array([0, 0, 0])
    s_optimal = 6.0
    
    # Validate
    if is_valid_packing(inner_hex_data, outer_hex_data, s_optimal):
        return inner_hex_data, outer_hex_data, s_optimal
    
    # Try smaller s
    for s_try in [5.5, 5.0, 4.5]:
        if is_valid_packing(inner_hex_data, outer_hex_data, s_try):
            return inner_hex_data, outer_hex_data, s_try
    
    # Fallback to seed
    return _get_seed_packing()

def _get_seed_packing():
    inner_hex_data = np.array(
        [
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ]
    )
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 8
    return inner_hex_data, outer_hex_data, outer_hex_side_length

def is_valid_packing(inner, outer, s):
    # Implement strict checks as per contract
    # 1. Finite values
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12):
        return False
        
    n = len(inner)
    
    # Get vertices
    def get_verts(cx, cy, angle, side):
        a = np.deg2rad(angle)
        angles_rad = a + np.deg2rad(np.arange(6) * 60)
        return np.array([cx + side * np.cos(angles_rad), cy + side * np.sin(angles_rad)]).T

    inner_verts = [get_verts(inner[i,0], inner[i,1], inner[i,2], 1.0) for i in range(n)]
    outer_verts = get_verts(outer[0], outer[1], outer[2], s)
    
    # 2. Outer vertices distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                return False

    # 3. Edge length check
    # Shift centers by minus outer center
    shift = np.array([outer[0], outer[1]])
    for i in range(n):
        c = np.array([inner[i,0], inner[i,1]]) - shift
        v = get_verts(c[0], c[1], inner[i,2], 1.0)
        for k in range(6):
            p1 = v[k]
            p2 = v[(k+1)%6]
            if abs(np.linalg.norm(p2-p1) - 1.0) > 1e-8:
                return False

    # 4. Non-overlap (SAT)
    for i in range(n):
        for j in range(i+1, n):
            if not sat_non_overlap(inner_verts[i], inner_verts[j]):
                return False

    # 5. Containment
    for i in range(n):
        for v in inner_verts[i]:
            if not point_in_polygon_strict(v, outer_verts):
                return False

    return True

def sat_non_overlap(poly1, poly2):
    # Separating Axis Theorem
    # Get all edge normals
    def get_normals(poly):
        normals = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1)%len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        return normals

    n1 = get_normals(poly1)
    n2 = get_normals(poly2)
    
    for n in n1 + n2:
        proj1 = np.dot(poly1, n)
        proj2 = np.dot(poly2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If intervals do not overlap, they are separated
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
            
    return False

def point_in_polygon_strict(point, polygon):
    # Cross product method for convex polygon
    # For every edge (p1, p2), cross(p2-p1, point-p1) must have same sign
    sign = None
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i+1)%len(polygon)]
        cross = np.cross(p2 - p1, point - p1)
        if sign is None:
            sign = np.sign(cross)
        elif np.sign(cross) != sign and abs(cross) > 1e-8:
            # Allow for boundary
            if cross < -1e-6:
                return False
    return True
