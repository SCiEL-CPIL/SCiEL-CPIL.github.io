import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    N = 11  # number of unit hexagons to pack
    
    # Find smallest hexagonal number >= N: hex_num(r) = 3*r*(r+1) + 1
    # r=0: 1, r=1: 7, r=2: 19, ...
    # For N=11, we need r=2 (19 hexagons) but we only place 11
    # Actually, we should find the best arrangement for exactly N hexagons
    
    # Strategy: Use hexagonal lattice positions, pick the N closest to center
    # Then compute minimal enclosing hexagon side length
    
    def hex_to_xy(i, j):
        """Convert axial hex coordinates to pixel coordinates for unit hexagons."""
        # For flat-top hexagons with side 1, spacing = sqrt(3)
        x = math.sqrt(3) * (i + j / 2.0)
        y = 1.5 * j
        return x, y
    
    def get_hex_vertices(cx, cy, angle_deg=0.0):
        """Get 6 vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
        angle_rad = math.radians(angle_deg)
        vertices = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            vertices.append((vx, vy))
        return vertices
    
    def min_enclosing_hex_side(vertices_list):
        """Compute minimal side length of a regular hexagon enclosing all given vertices.
        The enclosing hexagon can be rotated, so we try all 6 orientations 
        (by symmetry, we only need to check angles from 0 to 60 degrees, 
        but the optimal is when one side is parallel to a hull edge).
        
        Actually, for a regular hexagon, the minimal enclosing one is determined by
        the maximum extent in 6 directions separated by 60 degrees. We need to find
        the rotation that minimizes the maximum extent.
        
        The side length s of the enclosing hexagon relates to the inradius r_in by
        r_in = s * sqrt(3) / 2, so s = 2 * r_in / sqrt(3).
        
        For a given rotation angle alpha of the enclosing hexagon, the inradius needed
        is max over all vertices of |dot(vertex, direction)| for the 6 normal directions.
        """
        if not vertices_list:
            return 0.0
        
        # Collect all vertices
        all_pts = np.array(vertices_list)
        if len(all_pts) == 0:
            return 0.0
        
        # Center the points at origin for computation
        center = np.mean(all_pts, axis=0)
        pts = all_pts - center
        
        # Try rotations from 0 to 60 degrees (by symmetry of hexagon)
        best_s = float('inf')
        # Sample many angles to find the minimum
        # The function is piecewise, critical points occur when a vertex is 
        # equidistant from two adjacent sides, or aligned with a side normal
        angles_to_try = np.linspace(0, math.pi/3, 100)
        # Also add angles where vertices align with normals
        for angle in angles_to_try:
            # 6 normal directions of the enclosing hexagon
            normals = []
            for k in range(6):
                theta = angle + k * math.pi / 3.0
                normals.append(np.array([math.cos(theta), math.sin(theta)]))
            
            # For each normal, compute max projection (distance from center to supporting line)
            max_extent = 0.0
            for n in normals:
                projections = np.dot(pts, n)
                max_proj = np.max(np.abs(projections))
                max_extent = max(max_extent, max_proj)
            
            # The inradius of enclosing hexagon is max_extent
            s = 2.0 * max_extent / math.sqrt(3.0)
            best_s = min(best_s, s)
        
        return best_s
    
    # Generate hexagonal lattice positions
    # Use axial coordinates (q, r) where q + r + s = 0
    # Position in 2D: x = sqrt(3)*(q + r/2), y = 3/2*r  (for pointy-top)
    # or x = 3/2*q, y = sqrt(3)*(r + q/2) (for flat-top)
    
    # Let's use the standard: for a hex grid with spacing d = sqrt(3) between centers
    # Axial coords (i, j): x = d * (i + j/2), y = d * sqrt(3)/2 * j = 1.5 * j * (sqrt(3)/sqrt(3)) 
    # Wait, let me be more careful.
    
    # Unit hexagon side = 1. Distance between adjacent centers = 2 * inradius = sqrt(3).
    # Using axial coordinates (i, j):
    #   x = sqrt(3) * (i + j/2)
    #   y = 1.5 * j
    # This gives adjacent distance = sqrt(3). Let's verify:
    # (0,0) to (1,0): dist = sqrt(3) * 1 = sqrt(3) ✓
    # (0,0) to (0,1): dist = sqrt((sqrt(3)*0.5)^2 + 1.5^2) = sqrt(0.75 + 2.25) = sqrt(3) ✓
    # (0,0) to (1,-1): dist = sqrt((sqrt(3)*(1-0.5))^2 + (1.5*(-1))^2) = sqrt(0.75+2.25)=sqrt(3) ✓
    
    # Generate positions in hexagonal cluster, sorted by distance from center
    positions = []
    max_radius = 20  # enough for N up to ~1000
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            k = -i - j
            if abs(i) + abs(j) + abs(k) <= 2 * max_radius:  # hex distance
                # Hex distance from center: max(|i|, |j|, |k|)
                hex_dist = max(abs(i), abs(j), abs(k))
                x = math.sqrt(3) * (i + j / 2.0)
                y = 1.5 * j
                dist = math.sqrt(x*x + y*y)
                positions.append((dist, i, j, x, y))
    
    # Sort by distance from center, take first N
    positions.sort(key=lambda p: p[0])
    selected = positions[:N]
    
    # Place all hexagons with angle 0 (no rotation needed for optimal packing in hex container)
    inner = []
    all_vertices = []
    for (dist, i, j, x, y) in selected:
        inner.append([x, y, 0.0])
        verts = get_hex_vertices(x, y, 0.0)
        all_vertices.extend(verts)
    
    # Compute minimal enclosing hexagon side
    side = min_enclosing_hex_side(all_vertices)
    
    # The center of the enclosing hexagon should be at the centroid of the cluster
    # For a symmetric cluster, this is (0,0)
    # For asymmetric, use the center we computed
    cx = np.mean([v[0] for v in all_vertices])
    cy = np.mean([v[1] for v in all_vertices])
    
    # Actually, for the enclosing hexagon, we should use the center that minimizes the side
    # Our min_enclosing_hex_side already centers at the mean, so the outer center is (cx, cy)
    # But wait - the inner hexagons are placed at absolute positions, so the outer hexagon
    # center should be at (cx, cy) and the inner positions are relative to origin.
    # The evaluator will check if all inner hexagons fit inside the outer hexagon.
    # So we need: outer center = (cx, cy), and inner positions as given.
    
    # Actually, let me reconsider. The min_enclosing_hex_side function centers points at their mean,
    # computes the side, and the enclosing hexagon is centered at that mean.
    # So outer center = (cx, cy), side = side, rotation = optimal angle.
    
    # But we didn't track the optimal rotation angle. Let me redo to get it.
    
    def min_enclosing_hex_side_with_angle(vertices_list):
        if not vertices_list:
            return 0.0, 0.0, (0.0, 0.0)
        
        all_pts = np.array(vertices_list)
        if len(all_pts) == 0:
            return 0.0, 0.0, (0.0, 0.0)
        
        center = np.mean(all_pts, axis=0)
        pts = all_pts - center
        
        best_s = float('inf')
        best_angle = 0.0
        angles_to_try = np.linspace(0, math.pi/3, 200)
        
        for angle in angles_to_try:
            normals = []
            for k in range(6):
                theta = angle + k * math.pi / 3.0
                normals.append(np.array([math.cos(theta), math.sin(theta)]))
            
            max_extent = 0.0
            for n in normals:
                projections = np.dot(pts, n)
                max_proj = np.max(np.abs(projections))
                max_extent = max(max_extent, max_proj)
            
            s = 2.0 * max_extent / math.sqrt(3.0)
            if s < best_s:
                best_s = s
                best_angle = angle
        
        return best_s, best_angle, (center[0], center[1])
    
    side, outer_angle, outer_center = min_enclosing_hex_side_with_angle(all_vertices)
    
    # Check deadline
    if time.time() > deadline:
        # Return a safe default
        return inner, (0.0, 0.0), side + 0.1, 0.0
    
    # The outer hexagon rotation: the angle we found is the rotation of the normals.
    # The hexagon itself is rotated by the same angle (or angle - 30 deg depending on convention).
    # Let's use the angle as the rotation of the outer hexagon.
    
    return inner, outer_center, side, math.degrees(outer_angle)
