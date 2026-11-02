"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place 11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Strategy: Use a hexagonal lattice arrangement. The 11 hexagons are placed as:
    - 1 at center
    - 6 in first ring (hexagonal neighbors)
    - 4 in second ring (selected positions to keep compact)
    
    For unit hexagons (side length 1), the distance between centers of adjacent
    non-overlapping hexagons is 2 * inradius = 2 * (sqrt(3)/2) = sqrt(3).
    
    We compute the minimal regular hexagon that contains all vertices of all inner hexagons.
    """
    # Hexagonal lattice coordinates for 11 hexagons
    # Using axial coordinates (q, r) for hex grid
    # Center: (0,0)
    # First ring: 6 hexagons at distance 1 in hex grid
    # Second ring: pick 4 from the 12 positions to keep compact
    
    # Axial to pixel conversion for pointy-top hexagons
    # For pointy-top: x = size * sqrt(3) * (q + r/2), y = size * 3/2 * r
    # For flat-top: x = size * 3/2 * q, y = size * sqrt(3) * (r + q/2)
    
    # Let's use flat-top orientation. Size = 1 (side length = circumradius = 1)
    # Distance between adjacent centers = sqrt(3) * 1 = sqrt(3)
    
    hex_size = 1.0  # side length of inner hexagons
    sqrt3 = np.sqrt(3.0)
    
    # Axial coordinates for 11 hexagons in a compact arrangement
    # Center + 6 first ring + 4 second ring
    axial_coords = [
        (0, 0),      # center
        # First ring (6 hexagons)
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        # Second ring - pick 4 to make total 11, keeping symmetry
        (2, -1), (1, -2), (-1, 2), (-2, 1),
    ]
    
    # Convert axial to Cartesian (flat-top hexagons)
    # x = 3/2 * size * q
    # y = sqrt(3) * size * (r + q/2)
    centers = []
    for q, r in axial_coords:
        x = 1.5 * hex_size * q
        y = sqrt3 * hex_size * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # For each inner hexagon, compute its 6 vertices
    # A flat-top regular hexagon with circumradius R=1 has vertices at angles 0, 60, 120, 180, 240, 300
    # Vertex i: (cx + R*cos(theta_i), cy + R*sin(theta_i)) where theta_i = i*pi/3
    angles = np.array([i * np.pi / 3.0 for i in range(6)])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # All vertices of all inner hexagons
    all_vertices = []
    for cx, cy in centers:
        verts = np.column_stack([cx + cos_a, cy + sin_a])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    # Now find the minimal regular hexagon (any rotation) containing all these vertices
    # A regular hexagon with center (ox, oy), side length s, rotation angle alpha
    # has vertices at (ox + s*cos(alpha + k*pi/3), oy + s*sin(alpha + k*pi/3)) for k=0..5
    # 
    # A point (x,y) is inside the hexagon if for all 6 edges, the point is on the inside.
    # Equivalently, we can check the hexagonal distance.
    # 
    # For a regular hexagon with circumradius s centered at origin with flat top (alpha=0),
    # the condition for (x,y) to be inside is:
    # max(|x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2|) <= s * sqrt(3)/2
    # Wait, let me reconsider. For circumradius R, the inradius is R*sqrt(3)/2.
    # The hexagon can be described as the intersection of 6 half-planes.
    # 
    # Alternative: The minimum enclosing regular hexagon can be found by searching over
    # rotation angles and computing the required size for each.
    
    # For a given rotation angle alpha, the hexagon (centered at origin for now) 
    # contains a point p if the maximum over the 6 normal directions of the projection
    # is <= inradius = s * sqrt(3)/2.
    # 
    # The 6 outward normals for a flat-top hexagon rotated by alpha are at angles
    # alpha + k*pi/3 for k=0..5 (these are perpendicular to the edges).
    # Actually, for a regular hexagon, the normals to the edges are at angles
    # alpha + k*pi/3 where alpha is the angle to one of the edge normals.
    # 
    # Let me define: for a hexagon with circumradius s, the edge normals are at
    # angles beta, beta+pi/3, ..., beta+5*pi/3. The inradius is s*sqrt(3)/2.
    # A point p is inside iff |p · n_k| <= inradius for all k, where n_k are unit normals.
    # 
    # So for a given rotation (defined by normal angle beta), the required inradius is
    # max over all points p and all k of |p · n_k|, and s = inradius / (sqrt(3)/2).
    # 
    # We also need to find the optimal center. For a given rotation, the optimal center
    # that minimizes the maximum projection is related to the Chebyshev center in the
    # hexagonal metric. For simplicity, let's first try center = centroid of all vertices,
    # then optimize.
    
    # Compute centroid of all vertices
    centroid = np.mean(all_vertices, axis=0)
    
    # Try different rotation angles and find the minimal s
    best_s = float('inf')
    best_center = centroid
    best_angle = 0.0
    
    # Search over rotation angles
    # The hexagon has 6-fold symmetry, so we only need to search 0 to pi/3
    n_angles = 360
    for i in range(n_angles):
        alpha = i * np.pi / (3.0 * n_angles)  # 0 to pi/3
        # Normal angles
        normal_angles = alpha + np.arange(6) * np.pi / 3.0
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For this rotation, find the optimal center that minimizes the max projection
        # This is equivalent to finding the center of the minimal enclosing circle in
        # the dual sense. For a hexagon, the optimal center for a given orientation
        # can be found by solving a small optimization, but let's approximate.
        
        # Simple approach: use the centroid, then do a few iterations of optimization
        center = centroid.copy()
        
        # Compute required inradius for this center
        def compute_inradius(center, normals):
            # Project all vertices onto each normal
            rel = all_vertices - center
            projections = rel @ normals.T  # shape (n_points, 6)
            # For each normal, the max absolute projection
            max_proj = np.max(np.abs(projections), axis=0)  # shape (6,)
            # The inradius must be at least the max of these
            return np.max(max_proj)
        
        # Optimize center with a few gradient-free steps
        for iter in range(50):
            inr = compute_inradius(center, normals)
            # Compute gradient: for each point that achieves the max in some direction,
            # the center should move away from that point in that direction
            rel = all_vertices - center
            projections = rel @ normals.T
            abs_proj = np.abs(projections)
            max_per_normal = np.max(abs_proj, axis=0)
            inr = np.max(max_per_normal)
            
            # Find which (point, normal) pairs achieve the max
            # Move center to reduce the max projection
            # Simple subgradient: sum of unit vectors pointing from center to extreme points
            # weighted by which normal they're extreme in
            
            # For each normal k, find the point with max |projection|
            # and move center in the direction that reduces it
            grad = np.zeros(2)
            for k in range(6):
                n = normals[k]
                proj_k = projections[:, k]
                # Find point with max positive projection
                idx_pos = np.argmax(proj_k)
                # Find point with max negative projection
                idx_neg = np.argmin(proj_k)
                
                # If this normal determines the inradius, add gradient
                if abs(proj_k[idx_pos]) > inr - 1e-6 or abs(proj_k[idx_neg]) > inr - 1e-6:
                    # Move center toward the extreme point to reduce the projection
                    # Actually, we want to move center so that the projection decreases
                    # If point p has large positive projection on n, move center in direction n
                    # If point p has large negative projection on n, move center in direction -n
                    if proj_k[idx_pos] > inr - 1e-6:
                        grad += n
                    if proj_k[idx_neg] < -inr + 1e-6:
                        grad -= n
            
            if np.linalg.norm(grad) > 1e-10:
                # Step in the direction that reduces the max projection
                # We want to move center so that projections decrease
                # If grad points in direction d, moving center by -eta*d should help
                step = 0.1 / (1.0 + iter * 0.1)
                center -= step * grad / np.linalg.norm(grad)
        
        final_inr = compute_inradius(center, normals)
        s = final_inr / (np.sqrt(3.0) / 2.0)
        
        if s < best_s:
            best_s = s
            best_center = center
            best_angle = alpha
    
    # The outer hexagon has side length = circumradius = s
    # Return: inner hexagons (x, y, angle_degrees), outer center, outer side, outer angle
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    outer_center = (float(best_center[0]), float(best_center[1]))
    outer_side = float(best_s)
    outer_angle = float(best_angle)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
