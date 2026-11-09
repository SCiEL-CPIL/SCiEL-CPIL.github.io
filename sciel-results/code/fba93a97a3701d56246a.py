import numpy as np
import time
from itertools import product

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) side s rotated angle_deg."""
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point into hexagon's local frame
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # Regular hexagon with flat top: |x| <= s, |y| <= s*sqrt(3)/2, and |x|/s + |y|/(s*sqrt(3)/2) <= 1 for the hex shape
    # Actually for a regular hexagon with vertices at angles 0,60,...: the boundary is max of 6 half-planes
    # Half-plane i: cos(theta_i)*lx + sin(theta_i)*ly <= s * cos(pi/6) = s*sqrt(3)/2
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-12:
            return False
    return True

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    """Check if two hexagons (vertex arrays) are disjoint (no interior intersection)."""
    # Check if any vertex of one is inside the other
    for v in v1:
        if point_in_hex(v[0], v[1], 0, 0, s, 0) and False:
            pass
    # Better: check edge intersections and point-in-polygon
    # For simplicity, check if any vertex of v1 is strictly inside v2 and vice versa
    def point_in_poly(px, py, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for v in v1:
        if point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if point_in_poly(v[0], v[1], v1):
            return False
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j+1)%6]
            if seg_intersect(p1, p2, p3, p4):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Place 11 hexagons in a compact hexagonal cluster
    # Hex grid: center + 6 neighbors + 4 more
    # In hex grid coordinates, neighbors of (0,0) are at:
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1) in axial coords
    # Convert axial to pixel: x = s*sqrt(3)*(q + r/2), y = s*3/2*r  (for pointy-top)
    # For flat-top: x = s*3/2*q, y = s*sqrt(3)*(r + q/2)
    
    # Let's use flat-top hexagons. Side length = 1.
    # Distance between centers of edge-adjacent flat-top hexagons = sqrt(3)
    # Axial to pixel (flat-top): 
    #   x = 1.5 * q
    #   y = sqrt(3) * (r + 0.5*q)
    
    sqrt3 = np.sqrt(3)
    
    # 11 hexagons in hex pattern:
    # Layer 0: (0,0)
    # Layer 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Layer 2: pick 4 from the 12 positions at distance 2
    # Let's use a symmetric selection
    
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    # That's 11 positions
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # All inner hexagons have same orientation (0 degrees) and side 1
    inner_angles = [0.0] * N
    
    # Compute bounding box of all hexagon vertices to find good outer hex center
    all_verts = []
    for cx, cy in centers:
        verts = hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()
    
    # Try different outer hexagon rotations to minimize side length
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    
    # For a given outer hexagon rotation, the minimal side length to contain a set of points
    # is determined by the maximum "distance" in the 6 normal directions
    # For a regular hexagon with center (cx,cy), side s, rotation a:
    # The 6 face normals are at angles a + i*pi/3
    # A point (px,py) is inside if for all i: 
    #   (px-cx)*cos(n_i) + (py-cy)*sin(n_i) <= s * sqrt(3)/2
    # So s >= max over all points and all i of [(px-cx)*cos(n_i) + (py-cy)*sin(n_i)] / (sqrt(3)/2)
    
    # Coarse search over angles
    for outer_angle in np.linspace(0, 60, 121):  # 0 to 60 degrees
        a = np.radians(outer_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_verts[:, 0] - cx_mean
        dy = all_verts[:, 1] - cy_mean
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
    
    # Fine search around best angle
    if best_side < float('inf'):
        for outer_angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 201):
            a = np.radians(outer_angle)
            normals = []
            for i in range(6):
                theta = a + np.pi/6 + i * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            dx = all_verts[:, 0] - cx_mean
            dy = all_verts[:, 1] - cy_mean
            
            projections = np.zeros((N * 6, 6))
            for i in range(6):
                projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
            
            max_proj = projections.max(axis=0)
            min_proj = projections.min(axis=0)
            required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
            side = required_s.max()
            
            if side < best_side:
                best_side = side
                best_center = (cx_mean, cy_mean)
                best_angle = outer_angle
    
    # Optimize center for the best angle
    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + np.pi/6 + i * np.pi / 3), np.sin(a + np.pi/6 + i * np.pi / 3)] for i in range(6)])
    
    # The optimal center minimizes max over i of (max_proj_i - min_proj_i) / 2
    # where proj_i = dot(point - center, normal_i)
    # This is equivalent to: for each normal direction, the width is (max - min) of projections
    # The side length is max_i width_i / sqrt(3)
    # To minimize this, we want to center the point set optimally
    # For a given angle, the optimal center is the midpoint of the projection extremes in the "tightest" direction
    # Let's just try a few candidate centers
    
    # Candidate centers:
    # 1. Mean of all vertices
    # 2. Center of the bounding box in the global x/y frame
    # 3. Midpoint of the projection extents for the current normals (optimal for this angle)
    
    # Calculate optimal center for the current normals:
    # For each normal i, the projection of points is [min_i, max_i].
    # The center should be at (min_i + max_i) / 2 in that normal direction.
    # This is a 2D problem. We can just use the mean of the vertices as a good start,
    # or better, the center of the minimum enclosing circle/box.
    # For a hexagon, the "center" that minimizes max projection is the center of the
    # intersection of the strips defined by the normals.
    # A simple robust choice: the mean of the vertices is often close.
    # Let's try the mean, and the center of the axis-aligned bounding box.
    
    candidates = [
        (cx_mean, cy_mean),
        ((all_verts[:, 0].min() + all_verts[:, 0].max()) / 2, (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2)
    ]
    
    # Add the "projection midpoint" center:
    # For the current normals, the optimal center in 1D for normal i is (max_i + min_i)/2.
    # In 2D, we can't satisfy all 6 exactly, but we can try the average of these midpoints?
    # Actually, the center that minimizes the max width is the center of the smallest enclosing hexagon.
    # This is equivalent to the Chebyshev center.
    # For now, let's just evaluate the mean and the AABB center.
    
    for trial_cx, trial_cy in candidates:
        dx = all_verts[:, 0] - trial_cx
        dy = all_verts[:, 1] - trial_cy
        
        # Compute projections
        proj_matrix = np.dot(np.column_stack((dx, dy)), normals.T)
        max_proj = proj_matrix.max(axis=0)
        min_proj = proj_matrix.min(axis=0)
        
        # The required side length is determined by the max of the "half-widths"
        # half_width_i = max(max_proj_i, -min_proj_i)
        # side_i = half_width_i / (sqrt(3)/2)
        half_widths = np.maximum(max_proj, -min_proj)
        side = np.max(half_widths) / (sqrt3 / 2)
        
        if side < best_side:
            best_side = side
            best_center = (float(trial_cx), float(trial_cy))
    
    # Now verify that inner hexagons are actually disjoint
    # Check all pairs
    valid = True
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    for i in range(N):
        for j in range(i + 1, N):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: increase spacing slightly
        # Scale positions by a small factor
        scale = 1.001
        centers_scaled = centers * scale
        inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers_scaled]
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            all_verts = np.vstack(inner_verts)
            cx_mean = all_verts[:, 0].mean()
            cy_mean = all_verts[:, 1].mean()
            
            # Coarse search
            for outer_angle in np.linspace(0, 60, 121):
                a = np.radians(outer_angle)
                normals = []
                for i in range(6):
                    theta = a + np.pi/6 + i * np.pi / 3
                    normals.append([np.cos(theta), np.sin(theta)])
                normals = np.array(normals)
                
                dx = all_verts[:, 0] - cx_mean
                dy = all_verts[:, 1] - cy_mean
                
                projections = np.zeros((N * 6, 6))
                for i in range(6):
                    projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
                
                max_proj = projections.max(axis=0)
                min_proj = projections.min(axis=0)
                required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
                side = required_s.max()
                
                if side < best_side:
                    best_side = side
                    best_center = (cx_mean, cy_mean)
                    best_angle = outer_angle
            
            # Fine search
            if best_side < float('inf'):
                for outer_angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 201):
                    a = np.radians(outer_angle)
                    normals = []
                    for i in range(6):
                        theta = a + np.pi/6 + i * np.pi / 3
                        normals.append([np.cos(theta), np.sin(theta)])
                    normals = np.array(normals)
                    
                    dx = all_verts[:, 0] - cx_mean
                    dy = all_verts[:, 1] - cy_mean
                    
                    projections = np.zeros((N * 6, 6))
                    for i in range(6):
                        projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
                    
                    max_proj = projections.max(axis=0)
                    min_proj = projections.min(axis=0)
                    required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
                    side = required_s.max()
                    
                    if side < best_side:
                        best_side = side
                        best_center = (cx_mean, cy_mean)
                        best_angle = outer_angle
            centers = centers_scaled
    
    # Build inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
    
    # Final verification: ensure all inner hex vertices are inside outer hex
    # Use a tighter safety margin based on actual violation
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    all_check_verts = np.vstack(inner_verts)
    
    a = np.radians(best_angle)
    normals = []
    for i in range(6):
        theta = a + np.pi/6 + i * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    dx = all_check_verts[:, 0] - best_center[0]
    dy = all_check_verts[:, 1] - best_center[1]
    
    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_side * sqrt3 / 2)
        max_violation = max(max_violation, -proj.min() - best_side * sqrt3 / 2)
    
    if max_violation > 1e-9:
        # Add just enough to cover the violation, with a tiny epsilon for numerical safety
        best_side += max_violation / (sqrt3 / 2) + 1e-8
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
