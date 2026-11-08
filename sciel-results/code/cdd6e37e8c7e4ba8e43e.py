"""Pack 12 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 110.0  # safety margin before 120s limit

def _hex_verts(cx, cy, s, ang_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotated ang_deg."""
    ang = np.radians(ang_deg)
    # Regular hexagon vertices at angles 0, 60, ..., 300 degrees
    base_angles = np.arange(6) * np.pi / 3.0 + ang
    return np.column_stack([cx + s * np.cos(base_angles), cy + s * np.sin(base_angles)])

def _hex_face_normals(ang_deg):
    """Outward unit normals of a regular hexagon rotated by ang_deg.
    The hexagon has 6 faces; normals point outward at angles 30, 90, 150, 210, 270, 330 + ang."""
    ang = np.radians(ang_deg)
    base = np.radians(30) + np.arange(6) * np.pi / 3.0 + ang
    return np.column_stack([np.cos(base), np.sin(base)])

def _support(points, normals):
    """Support function: max dot product of points with each normal."""
    return points @ normals.T  # shape (n_points, n_normals)

def _min_enclosing_hex_side(points):
    """Given a set of 2D points, find the minimum side length of a regular hexagon
    (with arbitrary rotation and center) that contains all points.
    Returns (side_length, center_x, center_y, angle_deg)."""
    # A regular hexagon with side s, center c, rotation theta contains all points iff
    # for each of the 6 face normals n_i: max_j (p_j . n_i) - c . n_i <= s * cos(0) = s
    # Actually, the distance from center to each face is s (inradius = s for unit hex? No.)
    # For a regular hexagon with side length s: inradius = s * sqrt(3)/2, circumradius = s.
    # The distance from center to each side (apothem) = s * sqrt(3)/2.
    # So constraint: max_j (p_j . n_i) - c . n_i <= s * sqrt(3)/2 for all i.
    # And also: c . n_i - min_j (p_j . n_i) <= s * sqrt(3)/2 for all i.
    # Equivalently: (max_j (p_j . n_i) - min_j (p_j . n_i)) <= s * sqrt(3) for all i.
    # And s = max_i (max_j(p_j.n_i) - min_j(p_j.n_i)) / sqrt(3).
    # Center c can be chosen as the midpoint: c . n_i = (max + min)/2.
    
    # But we also need to optimize over rotation theta.
    # For a fixed theta, the optimal s is max over the 6 directions of (max - min) / sqrt(3).
    # We minimize over theta.
    
    apothem_factor = np.sqrt(3) / 2.0
    # Actually let me re-derive. For a regular hexagon with side s:
    # The 6 sides are at distance s*sqrt(3)/2 from center.
    # Normal direction for face i: angle = 30° + 60°*i + theta
    # Constraint: for face i with outward normal n_i:
    #   max_j (p_j . n_i) - c . n_i <= s * sqrt(3)/2
    #   c . n_i - min_j (p_j . n_i) <= s * sqrt(3)/2
    # Adding: max_j(p_j.n_i) - min_j(p_j.n_i) <= s * sqrt(3)
    # So s >= (max_j(p_j.n_i) - min_j(p_j.n_i)) / sqrt(3)
    # Optimal s for given theta: s(theta) = max_i (range_i(theta)) / sqrt(3)
    # where range_i = max_j(p_j . n_i(theta)) - min_j(p_j . n_i(theta))
    
    def s_of_theta(theta):
        ang = theta
        base = np.radians(30) + np.arange(6) * np.pi / 3.0 + ang
        normals = np.column_stack([np.cos(base), np.sin(base)])
        dots = points @ normals.T  # (n_pts, 6)
        ranges = dots.max(axis=0) - dots.min(axis=0)
        return np.max(ranges) / np.sqrt(3)
    
    # Minimize s_of_theta over theta in [0, 2*pi)
    # Use a grid + local optimization
    best_s = float('inf')
    best_theta = 0.0
    
    # Coarse grid
    n_grid = 72
    for i in range(n_grid):
        theta = 2 * np.pi * i / n_grid
        s = s_of_theta(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    # Refine with local optimization
    try:
        res = minimize(s_of_theta, [best_theta], method='Nelder-Mead',
                       options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200})
        if res.fun < best_s:
            best_s = res.fun
            best_theta = res.x[0]
    except:
        pass
    
    # Compute center
    ang = best_theta
    base = np.radians(30) + np.arange(6) * np.pi / 3.0 + ang
    normals = np.column_stack([np.cos(base), np.sin(base)])
    dots = points @ normals.T
    # Center projection on each normal: (max + min) / 2
    c_projs = (dots.max(axis=0) + dots.min(axis=0)) / 2.0
    # Solve for c: normals.T @ c = c_projs (6 equations, 2 unknowns) - least squares
    c, _, _, _ = np.linalg.lstsq(normals.T, c_projs, rcond=None)
    cx, cy = c[0], c[1]
    
    return best_s, cx, cy, np.degrees(best_theta) % 360.0

def _all_hex_points(centers, angles_deg):
    """Get all vertices of all hexagons as a single array (n_total_points, 2)."""
    all_pts = []
    for (cx, cy), ang in zip(centers, angles_deg):
        verts = _hex_verts(cx, cy, 1.0, ang)
        all_pts.append(verts)
    return np.vstack(all_pts)

def _objective(x):
    """x = [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11] (36 params)"""
    centers = np.array(x[0::3]).reshape(-1, 2)
    angles = x[1::3]
    pts = _all_hex_points(centers, angles)
    s, _, _, _ = _min_enclosing_hex_side(pts)
    return s

def _check_validity(centers, angles_deg, outer_center, outer_side, outer_angle):
    """Check that all inner hexagons are disjoint and inside the outer hexagon."""
    # Check containment: all vertices of inner hexagons must be inside outer hexagon
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    all_pts = _all_hex_points(centers, angles_deg)
    
    # Point-in-polygon test for convex polygon (outer hexagon)
    # For a convex polygon, point is inside iff it's on the same side of each edge
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        # Edge vector
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # Normal pointing inward
        nx, ny = -ey, ex
        # Check all points
        for pt in all_pts:
            cross = ex * (pt[1] - v1[1]) - ey * (pt[0] - v1[0])
            # All points should be on the same side (inside)
            # Determine which side is inside using center
            cross_center = ex * (outer_center[1] - v1[1]) - ey * (outer_center[0] - v1[0])
            if cross * cross_center < -1e-9:
                return False
    
    # Check disjointness: for each pair, check if any vertex of one is inside the other
    # and vice versa, and check edge intersections
    for i in range(N):
        verts_i = _hex_verts(centers[i][0], centers[i][1], 1.0, angles_deg[i])
        for j in range(i + 1, N):
            verts_j = _hex_verts(centers[j][0], centers[j][1], 1.0, angles_deg[j])
            # Check if any vertex of i is inside j
            for pt in verts_i:
                if _point_in_convex(pt, verts_j):
                    return False
            for pt in verts_j:
                if _point_in_convex(pt, verts_i):
                    return False
            # Check edge intersections
            if _edges_intersect(verts_i, verts_j):
                return False
    return True

def _point_in_convex(pt, polygon):
    """Check if point is strictly inside convex polygon (6 vertices)."""
    for i in range(len(polygon)):
        v1 = polygon[i]
        v2 = polygon[(i + 1) % len(polygon)]
        cross = (v2[0] - v1[0]) * (pt[1] - v1[1]) - (v2[1] - v1[1]) * (pt[0] - v1[0])
        if cross < -1e-9:
            return False
    return True

def _edges_intersect(poly1, poly2):
    """Check if any edge of poly1 intersects any edge of poly2."""
    for i in range(len(poly1)):
        a1 = poly1[i]
        b1 = poly1[(i + 1) % len(poly1)]
        for j in range(len(poly2)):
            a2 = poly2[j]
            b2 = poly2[(j + 1) % len(poly2)]
            if _seg_intersect(a1, b1, a2, b2):
                return True
    return False

def _seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    return False

def construct():
    t_start = time.time()
    
    # Initial guess: place hexagons in a ring pattern
    # Try several initial configurations
    best_result = None
    best_side = float('inf')
    
    # Configuration 1: ring of 12 around center
    # Configuration 2: two rows
    # Configuration 3: hexagonal packing
    
    # Let's try a structured initial placement
    # Hexagons in a honeycomb-like arrangement
    # For 12 hexagons, a 4x3 grid in hexagonal coordinates works well
    
    # Hex grid coordinates to Cartesian
    def hex_to_cart(q, r, size=1.0):
        x = size * (q + r / 2.0) * 2.0 * np.sqrt(3) / 2.0  # axial to pixel
        # Actually for pointy-top hexagons with side s:
        # x = s * sqrt(3) * (q + r/2)
        # y = s * 3/2 * r
        x = size * np.sqrt(3) * (q + r / 2.0)
        y = size * 1.5 * r
        return x, y
    
    # Place 12 hexagons in a hexagonal grid pattern
    # A cluster of 12: center + 6 around + 5 more
    # Or just a 4x3 rectangular grid in hex coords
    positions = []
    # 3 rows of 4
    for row in range(3):
        for col in range(4):
            q = col - 1.5  # center the grid
            r = row - 1
            x, y = hex_to_cart(q, r)
            positions.append((x, y))
    
    # Convert to parameter vector
    x0 = np.zeros(3 * N)
    for i, (px, py) in enumerate(positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = 0.0  # no rotation initially
    
    # Optimize
    try:
        res = minimize(_objective, x0, method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 500,
                                'adaptive': True})
        
        if time.time() - t_start < DEADLINE:
            x_opt = res.x
            centers = np.array(x_opt[0::3]).reshape(-1, 2)
            angles = x_opt[1::3]
            pts = _all_hex_points(centers, angles)
            side, cx, cy, ang = _min_enclosing_hex_side(pts)
            
            # Verify validity
            if _check_validity(centers, angles, (cx, cy), side, ang):
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles, (cx, cy), side, ang)
    except Exception as e:
        pass
    
    # Try another initial configuration if time permits
    if time.time() - t_start < DEADLINE - 20:
        # Try with some rotation
        x1 = x0.copy()
        for i in range(N):
            x1[3*i+2] = 30.0  # 30 degree rotation
        
        try:
            res2 = minimize(_objective, x1, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 300,
                                    'adaptive': True})
            if res2.fun < (best_side if best_result else float('inf')):
                x_opt = res2.x
                centers = np.array(x_opt[0::3]).reshape(-1, 2)
                angles = x_opt[1::3]
                pts = _all_hex_points(centers, angles)
                side, cx, cy, ang = _min_enclosing_hex_side(pts)
                if _check_validity(centers, angles, (cx, cy), side, ang):
                    if side < best_side:
                        best_side = side
                        best_result = (centers, angles, (cx, cy), side, ang)
        except:
            pass
    
    # If no valid result found, use a safe fallback
    if best_result is None:
        # Fallback: spread out hexagons in a large hexagon
        positions = []
        for row in range(3):
            for col in range(4):
                q = col - 1.5
                r = row - 1
                x, y = hex_to_cart(q, r, size=2.0)  # larger spacing
                positions.append((x, y))
        centers = np.array(positions)
        angles = np.zeros(N)
        pts = _all_hex_points(centers, angles)
        side, cx, cy, ang = _min_enclosing_hex_side(pts)
        best_result = (centers, angles, (cx, cy), side, ang)
        best_side = side
    
    centers, angles, (cx, cy), side, ang = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(cx), float(cy)), float(side), float(ang)
