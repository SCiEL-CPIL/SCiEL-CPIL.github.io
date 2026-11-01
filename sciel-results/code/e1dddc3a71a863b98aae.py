import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Helper to get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def hex_vertices(cx, cy, angle_deg=0.0):
        angle = np.deg2rad(angle_deg)
        # Regular hexagon side length 1
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees relative to center
        # Flat-top orientation usually means vertices at 0, 60... or 30, 90...
        # Standard math definition: vertex at (1, 0) is angle 0.
        # Let's use pointy-top (vertex at 90 deg) or flat-top?
        # "Regular hexagon" usually implies symmetry.
        # Let's define a standard hexagon.
        # If we align with axes, it's easier.
        # Let's use the standard "flat-top" hexagon where one side is horizontal.
        # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # This has width 2, height sqrt(3).
        # Rotation angle 0 corresponds to this orientation.
        base_vertices = np.array([
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1.0, 0.0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Rotate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        rotated = base_vertices @ rot_matrix.T
        return rotated + np.array([cx, cy])

    def min_dist_polygons(v1, v2):
        """
        v1, v2: arrays of shape (6, 2)
        Returns minimum distance between any vertex in v1 and any vertex in v2,
        or 0 if they intersect/overlap (approximated by min vertex-vertex dist, 
        but for convex polygons, if min vertex-vertex dist is small, they might still overlap.
        Better: check if any vertex of v1 is inside v2 or vice versa, or if edges cross.
        For simplicity and speed in heuristic, we'll use a robust check:
        1. Check if any vertex of v1 is strictly inside v2.
        2. Check if any vertex of v2 is strictly inside v1.
        3. Check for edge intersections.
        If none, min distance is min vertex-vertex distance.
        """
        # This is computationally heavy if done naively for 12x12 pairs in a loop.
        # We will rely on the fact that in tight packings, contact is often vertex-vertex or edge-edge.
        # For a robust "no overlap" check, we need to be sure.
        
        # Point in polygon check (ray casting or winding)
        def point_in_polygon(pt, poly):
            # poly is (N, 2)
            x, y = pt
            inside = False
            n = len(poly)
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        # Edge intersection check
        def segments_intersect(p1, p2, p3, p4):
            # Check if segment p1-p2 intersects p3-p4
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        # Check overlaps
        for i in range(6):
            if point_in_polygon(v1[i], v2):
                return 0.0
            if point_in_polygon(v2[i], v1):
                return 0.0
        
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return 0.0
        
        # If no overlap, return min vertex-vertex distance
        min_d = float('inf')
        for i in range(6):
            for j in range(6):
                d = np.linalg.norm(v1[i] - v2[j])
                if d < min_d:
                    min_d = d
        return min_d

    def required_outer_side(inner_positions, inner_angles, outer_center, outer_angle_deg=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at outer_center
        with rotation outer_angle_deg that contains all inner hexagons.
        """
        oc = np.array(outer_center)
        oa = np.deg2rad(outer_angle_deg)
        
        # Outer hexagon normal directions
        # For a flat-top hexagon, normals are at 0, 60, 120, 180, 240, 300 degrees
        # The distance from center to side (apothem) is R * cos(30) = R * sqrt(3)/2
        # Side length s = 2 * R * sin(30) = R. So R = s.
        # Distance from center to side = s * sqrt(3)/2.
        
        # We need to find the maximum projection of all inner vertices onto the 6 normal directions.
        # Let n_k be the unit normal vector for side k.
        # The constraint is: dot(v - center, n_k) <= apothem for all vertices v and all k.
        # apothem = s * sqrt(3)/2.
        # So s >= max_k max_v dot(v - center, n_k) / (sqrt(3)/2)
        
        normals = []
        for k in range(6):
            angle = oa + np.deg2rad(k * 60)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            ang = inner_angles[i]
            verts = hex_vertices(cx, cy, ang)
            for v in verts:
                rel_v = v - oc
                for n in normals:
                    proj = np.dot(rel_v, n)
                    if proj > max_proj:
                        max_proj = proj
        
        if max_proj <= 0:
            return 0.0
        
        apothem_required = max_proj
        side_length = apothem_required / (np.sqrt(3) / 2)
        return side_length

    def check_validity(inner_positions, inner_angles):
        """
        Check if inner hexagons are disjoint.
        """
        verts_list = []
        for i in range(len(inner_positions)):
            cx, cy = inner_positions[i]
            ang = inner_angles[i]
            verts_list.append(hex_vertices(cx, cy, ang))
        
        for i in range(len(verts_list)):
            for j in range(i + 1, len(verts_list)):
                d = min_dist_polygons(verts_list[i], verts_list[j])
                if d < 1e-6: # Allow tiny numerical error
                    return False
        return True

    # Strategy: Try specific structured configurations
    # 1. 3x4 Grid (Rectangular block)
    # 2. Hexagonal Cluster (1 center, 6 ring, 5 more?)
    # 3. Honeycomb strips
    
    best_config = None
    best_score = float('inf')
    
    # Configuration 1: 3 rows x 4 cols grid
    # Unit hexagons in a grid:
    # Flat-top hexagons tile the plane.
    # Horizontal distance between centers: 1.5 * side = 1.5
    # Vertical distance between centers: sqrt(3) * side = sqrt(3)
    # Offset for alternate rows: 0.75 * side = 0.75
    
    # Let's generate a 3x4 grid
    pos_3x4 = []
    angles_3x4 = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_3x4.append((x, y))
            angles_3x4.append(0.0)
    
    # Center the grid
    cx_grid = np.mean([p[0] for p in pos_3x4])
    cy_grid = np.mean([p[1] for p in pos_3x4])
    pos_3x4 = [(x - cx_grid, y - cy_grid) for x, y in pos_3x4]
    
    if check_validity(pos_3x4, angles_3x4):
        s = required_outer_side(pos_3x4, angles_3x4, (0,0), 0.0)
        if s < best_score:
            best_score = s
            best_config = (pos_3x4, angles_3x4, (0.0, 0.0), 0.0)
    
    # Configuration 2: Hexagonal cluster (Center + 6 neighbors + 5 more in second ring?)
    # A perfect hexagonal cluster of 19 hexagons (radius 2) is too big.
    # A cluster of 7 (radius 1) is small.
    # Let's try a "flower" shape: 1 center, 6 around. That's 7.
    # We need 12. Add 5 more?
    # Maybe two clusters of 6? Or a 2x6 strip?
    
    # Configuration 3: 2 rows x 6 cols
    pos_2x6 = []
    angles_2x6 = []
    for r in range(2):
        for c in range(6):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_2x6.append((x, y))
            angles_2x6.append(0.0)
            
    cx_grid = np.mean([p[0] for p in pos_2x6])
    cy_grid = np.mean([p[1] for p in pos_2x6])
    pos_2x6 = [(x - cx_grid, y - cy_grid) for x, y in pos_2x6]
    
    if check_validity(pos_2x6, angles_2x6):
        s = required_outer_side(pos_2x6, angles_2x6, (0,0), 0.0)
        if s < best_score:
            best_score = s
            best_config = (pos_2x6, angles_2x6, (0.0, 0.0), 0.0)
            
    # Configuration 4: 4 rows x 3 cols
    pos_4x3 = []
    angles_4x3 = []
    for r in range(4):
        for c in range(3):
            x = c * 1.5
            y = r * np.sqrt(3)
            if r % 2 == 1:
                x += 0.75
            pos_4x3.append((x, y))
            angles_4x3.append(0.0)
            
    cx_grid = np.mean([p[0] for p in pos_4x3])
    cy_grid = np.mean([p[1] for p in pos_4x3])
    pos_4x3 = [(x - cx_grid, y - cy_grid) for x, y in pos_4x3]
    
    if check_validity(pos_4x3, angles_4x3):
        s = required_outer_side(pos_4x3, angles_4x3, (0,0), 0.0)
        if s < best_score:
            best_score = s
            best_config = (pos_4x3, angles_4x3, (0.0, 0.0), 0.0)

    # Configuration 5: Optimized Hexagonal Lattice Subset
    # Take a large hexagonal lattice and pick 12 points that are close together.
    # This is essentially finding the densest packing of 12 points in a hex lattice.
    # The "hexagonal number" 12 is not a perfect hexagon (1, 7, 19).
    # So we take a 7-cluster and add 5 from the next ring.
    
    # Generate hex lattice points
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Axial to pixel (flat-top)
            x = 1.5 * q
            y = np.sqrt(3) * (r + q/2.0)
            lattice_points.append((x, y))
    
    # Sort by distance from origin
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Try subsets of 12 from the first 20 closest points
    # This is combinatorial explosion. Instead, try a greedy addition or specific patterns.
    # Let's just try the 12 closest points to the origin in the lattice.
    # The 12 closest points to (0,0) in a hex lattice:
    # (0,0) is 1.
    # 6 neighbors are at dist 1.5 (approx? No, dist is sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25+0.75)=sqrt(3)? No.
    # Distance between (0,0) and (1,0) in axial: x=1.5, y=sqrt(3)/2. dist = sqrt(2.25 + 0.75) = sqrt(3) ~ 1.732.
    # Wait, side length is 1. Distance between centers of touching hexagons is sqrt(3) for flat-top?
    # Let's re-verify.
    # Flat-top hex, side 1.
    # Center to vertex: 1.
    # Center to edge midpoint (apothem): sqrt(3)/2.
    # Distance between centers of two touching flat-top hexagons horizontally:
    # One is at (0,0). Next one to the right touches at x = 1 + 1 = 2? No.
    # Vertices at (1,0), (0.5, sqrt(3)/2)...
    # The rightmost point is (1,0). The leftmost point of the next one is (-1,0) relative to its center.
    # So centers are separated by 2? No, they share an edge?
    # Flat-top hexagons do not share vertical edges. They share slanted edges.
    # The horizontal distance between centers in a row is 1.5 * side?
    # Let's check: x-coords of vertices: 1, 0.5, -0.5, -1, -0.5, 0.5.
    # Width is 2.
    # If we place another one to the right, its leftmost vertex is at x_center - 1.
    # For them to touch, x_center1 + 1 = x_center2 - 1? No, they touch at an edge.
    # The edge is between (0.5, sqrt(3)/2) and (0.5, -sqrt(3)/2)? No, that's a vertical line segment?
    # Vertices:
    # 0: (1, 0)
    # 1: (0.5, sqrt(3)/2)
    # 2: (-0.5, sqrt(3)/2)
    # 3: (-1, 0)
    # 4: (-0.5, -sqrt(3)/2)
    # 5: (0.5, -sqrt(3)/2)
    # Edge 0-1: (1,0) to (0.5, sqrt(3)/2).
    # Edge 5-0: (0.5, -sqrt(3)/2) to (1,0).
    # These are the right-most edges.
    # If we place another hex at (x2, 0), its left-most edges are 2-3 and 3-4.
    # Vertex 3 is at (x2-1, 0). Vertex 2 is at (x2-0.5, sqrt(3)/2).
    # For edge 0-1 of hex1 to touch edge 2-3 of hex2:
    # Hex1 edge 0-1 midpoint: (0.75, sqrt(3)/4).
    # Hex2 edge 2-3 midpoint: (x2-0.75, sqrt(3)/4).
    # They are collinear if y is same.
    # Distance between midpoints: x2 - 0.75 - 0.75 = x2 - 1.5.
    # For them to touch, the distance should be 0? No, they are parallel edges.
    # The distance between the lines containing the edges is the gap.
    # Actually, in a hex grid, the center distance is sqrt(3) * side?
    # Let's calculate distance between (0,0) and (1.5, sqrt(3)/2).
    # dx = 1.5, dy = sqrt(3)/2.
    # dist^2 = 2.25 + 0.75 = 3.0. dist = sqrt(3).
    # Yes, center distance is sqrt(3) for touching hexagons in a flat-top grid.
    
    # So the lattice points I generated with x=1.5*q, y=sqrt(3)*(r+q/2) are correct for center positions.
    # The 12 closest points to origin:
    # (0,0) -> 1
    # 6 neighbors at dist sqrt(3) -> 6
    # 6 more neighbors at dist 3? (e.g. (3,0), (1.5, 1.5sqrt(3))?)
    # Let's just take the 12 closest lattice points.
    
    closest_12 = lattice_points[:12]
    
    # Center them
    cx_c = np.mean([p[0] for p in closest_12])
    cy_c = np.mean([p[1] for p in closest_12])
    pos_c = [(x - cx_c, y - cy_c) for x, y in closest_12]
    
    if check_validity(pos_c, [0.0]*12):
        s = required_outer_side(pos_c, [0.0]*12, (0,0), 0.0)
        if s < best_score:
            best_score = s
            best_config = (pos_c, [0.0]*12, (0.0, 0.0), 0.0)

    if best_config is None:
        # Fallback to a safe, large configuration
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    # Refine the best config using L-BFGS-B
    # Variables: x1, y1, a1, ..., x12, y12, a12
    def objective(vars):
        if time.time() > deadline:
            return 1e10
        pos = [(vars[i*3], vars[i*3+1]) for i in range(N)]
        ang = [vars[i*3+2] for i in range(N)]
        
        # Check validity
        if not check_validity(pos, ang):
            return 1e10
        
        s = required_outer_side(pos, ang, (0,0), 0.0)
        return s

    x0 = []
    for i in range(N):
        x0.append(best_config[0][i][0])
        x0.append(best_config[0][i][1])
        x0.append(best_config[1][i])
        
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'disp': False})
        if res.success or (not res.success and res.fun < best_score):
            if time.time() < deadline:
                pos_opt = [(res.x[i*3], res.x[i*3+1]) for i in range(N)]
                ang_opt = [res.x[i*3+2] for i in range(N)]
                if check_validity(pos_opt, ang_opt):
                    s_opt = required_outer_side(pos_opt, ang_opt, (0,0), 0.0)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_config = (pos_opt, ang_opt, (0.0, 0.0), 0.0)
    except Exception:
        pass

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_config[0][i][0], best_config[0][i][1], np.rad2deg(best_config[1][i])])
    
    return inner, best_config[2], best_score, best_config[3]
# EVOLVE-BLOCK-END
