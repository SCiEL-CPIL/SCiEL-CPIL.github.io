import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 12 centers for unit hexagons (flat-top orientation)
    # Distance between adjacent centers is sqrt(3)
    d = np.sqrt(3)
    
    # 7 central hexagons
    centers = np.array([
        [0.0, 0.0],
        [d, 0.0],
        [d/2, 1.5],
        [-d/2, 1.5],
        [-d, 0.0],
        [-d/2, -1.5],
        [d/2, -1.5]
    ])
    
    # 5 additional hexagons from the next layer (distance 3 from center)
    # Axial coords (1,1), (1,-2), (-2,1), (-1,-1), (-2,-1) ? 
    # Let's pick 5 of the 6 points at distance 3.
    # Points at dist 3 in Cartesian:
    # (1,1) -> (1.5d, 1.5)
    # (1,-2) -> (1.5d, -1.5) ... wait
    # Axial (q,r) to Cartesian: x = d(q + r/2), y = 1.5r
    # (1,1): x = 1.5d, y = 1.5
    # (1,-2): x = d(1-1)=0, y = -3
    # (-2,1): x = d(-2+0.5) = -1.5d, y = 1.5
    # (-1,-1): x = d(-1-0.5) = -1.5d, y = -1.5
    # (-2,-1): x = d(-2-0.5) = -2.5d, y = -1.5  -- Wait, dist check
    # Let's just use the 6 points and drop one.
    outer_pts = np.array([
        [1.5*d, 1.5],   # (1,1)
        [0.0, -3.0],    # (1,-2) -> x=0, y=-3
        [-1.5*d, 1.5],  # (-2,1)
        [-1.5*d, -1.5], # (-1,-1)
        [-2.5*d, -1.5], # (-2,-1) -> x=-2.5d? No.
        [2.5*d, -1.5]   # (2,-1)
    ])
    # Correct Cartesian for axial (q,r):
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # (1,1): x=1.5sqrt3, y=1.5
    # (1,-2): x=sqrt3(1-1)=0, y=-3
    # (-2,1): x=sqrt3(-2+0.5)=-1.5sqrt3, y=1.5
    # (-1,-1): x=sqrt3(-1-0.5)=-1.5sqrt3, y=-1.5
    # (-2,-1): x=sqrt3(-2-0.5)=-2.5sqrt3, y=-1.5  (Dist sqrt(18.75+2.25)=sqrt(21)~4.58? No)
    # Let's re-calculate dist for (-2,-1): q=-2, r=-1.
    # q^2+qr+r^2 = 4+2+1=7. Dist = sqrt(3)*sqrt(7) = sqrt(21) ~ 4.58.
    # My previous "dist 3" was for q^2+qr+r^2=3.
    # (1,1): 1+1+1=3. Dist 3.
    # (1,-2): 1-2+4=3. Dist 3.
    # (-2,1): 4-2+1=3. Dist 3.
    # (-1,-1): 1+1+1=3. Dist 3.
    # (-1,2): 1-2+4=3. Dist 3.
    # (2,-1): 4-2+1=3. Dist 3.
    
    # So the 6 points are:
    # (1,1), (1,-2), (-2,1), (-1,-1), (-1,2), (2,-1)
    
    pts_layer2 = np.array([
        [1.5*d, 1.5],   # (1,1)
        [0.0, -3.0],    # (1,-2)
        [-1.5*d, 1.5],  # (-2,1)
        [-1.5*d, -1.5], # (-1,-1)
        [0.0, 3.0],     # (-1,2) -> x=sqrt3(-1+1)=0, y=3
        [1.5*d, -1.5]   # (2,-1)
    ])
    
    # Take 5 of them to make 12 total.
    # Drop one to break symmetry slightly, but keep it compact.
    # Dropping (1,1) for example.
    selected_outer = pts_layer2[1:] # Drop first one
    
    all_centers = np.vstack((centers, selected_outer))
    
    # Center the cluster
    centroid = np.mean(all_centers, axis=0)
    all_centers = all_centers - centroid
    
    # Find minimal enclosing regular hexagon side length
    # The side length S is determined by the max projection onto the 6 normals.
    # S = max_k |p . n_k| * 2/sqrt(3) + 1 (for unit hexagons)
    # We can rotate the cluster to minimize this max projection.
    
    min_side = float('inf')
    # Try rotations
    for angle_deg in np.linspace(0, 60, 121):
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        rotated_centers = all_centers @ rot_matrix.T
        
        # Normals for hexagon aligned with axes (flat top)
        # Normals at 0, 60, 120, 180, 240, 300
        normals = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ])
        
        # Project centers onto normals
        # For each normal, find max and min projection
        # The width in that direction is max - min
        # The apothem required is max(width / 2, max_abs_proj) ?
        # No, the hexagon is centered at origin.
        # Condition: |p . n_k| <= A for all p, k.
        # So A >= max_k max_p |p . n_k|
        
        projections = rotated_centers @ normals.T
        abs_projections = np.abs(projections)
        max_abs_proj = np.max(abs_projections)
        
        # Apothem A
        A = max_abs_proj
        # Side length of outer hexagon
        S = A * 2 / np.sqrt(3) + 1.0
        
        if S < min_side:
            min_side = S
            best_angle = angle_deg
    
    # Return the best configuration
    # We need to return inner (x, y, angle), center, side, angle
    # Inner hexagons are aligned with outer (angle 0) in this calculation.
    # But we rotated the cluster to find the optimal orientation.
    # So we should return the rotated centers and the outer hexagon angle.
    
    # Re-rotate back to get the coordinates relative to the outer hexagon?
    # No, the return format is:
    # inner: rows (x, y, angle_degrees) of n unit hexagons
    # outer: center (x, y), side length, rotation
    
    # We found the optimal rotation for the cluster relative to the fixed normals.
    # This is equivalent to fixing the cluster and rotating the normals (outer hexagon).
    # So the outer hexagon angle is -best_angle?
    # Let's say we rotated the cluster by `best_angle`.
    # The normals were fixed.
    # So the outer hexagon is aligned with the fixed normals.
    # The inner hexagons are rotated by `best_angle`.
    # But we can also just return the original centers and set the outer hexagon angle to `-best_angle`.
    
    # Let's return the original centers (unrotated) and the outer hexagon rotated by -best_angle.
    # And the inner hexagons can be at angle 0 (aligned with their local frame).
    # Wait, if I rotate the outer hexagon, the "flat top" orientation of the inner hexagons
    # relative to the outer hexagon changes.
    # The task allows inner hexagons to rotate.
    # So I can set inner angle to 0, and outer angle to -best_angle.
    
    # The centers returned should be in the global frame.
    # If I return the unrotated centers, and the outer hexagon is rotated,
    # the check will pass if the geometry is correct.
    
    inner = []
    for x, y in all_centers + centroid: # Restore centroid? 
        # The construct function returns absolute coordinates.
        # The evaluator will place them.
        # It's best to return them centered at (0,0) for simplicity,
        # and set outer center to (0,0).
        pass
        
    # Let's center them at 0,0
    inner = [[x, y, 0.0] for x, y in all_centers]
    center = (0.0, 0.0)
    side = min_side
    angle = -best_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
