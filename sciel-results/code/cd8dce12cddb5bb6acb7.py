import numpy as np

N = 12

def construct():
    # Generate centers for 12 hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    
    # 6 in Ring 1
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.deg2rad(60 * i)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
        
    # 5 in Ring 2
    # We need to choose 5 out of 12 positions in Ring 2.
    # Let's try 5 contiguous ones.
    r2 = 2 * np.sqrt(3)
    # Ring 2 positions are at angles 0, 30, 60... ? No.
    # Ring 2 neighbors of Ring 1.
    # A Ring 1 hex at angle 0 has neighbors at angles 60 and -60 relative to it?
    # Let's just generate all 12 Ring 2 positions and pick 5.
    ring2_centers = []
    for i in range(6):
        angle = np.deg2rad(60 * i)
        cx = r1 * np.cos(angle)
        cy = r1 * np.sin(angle)
        # Neighbors of this Ring 1 hex that are further from origin
        # The "outer" neighbors are at angles angle +/- 60 deg from the vector to the hex?
        # Actually, the lattice vectors are e1, e2.
        # Simpler: The Ring 2 centers are at distance 2*sqrt(3) from origin?
        # No, distance from origin to Ring 2 center is 2*sqrt(3) only for the "corners" of the hexagon.
        # Let's use axial coordinates.
        
    # Let's use axial coordinates (q, r) for hex grid.
    # Center (0,0).
    # Neighbors: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1).
    # Let's generate all hexes with distance <= 2 from center.
    # Distance in hex grid is max(|q|, |r|, |q+r|)? No, (|q|+|r|+|q+r|)/2.
    
    hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check distance
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= 2:
                hexes.append((q, r))
                
    # Convert axial to pixel
    # x = size * sqrt(3) * (q + r/2)
    # y = size * 3/2 * r
    # size = 1
    pixel_centers = []
    for q, r in hexes:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pixel_centers.append((x, y))
        
    # We have 19 hexes. We need 12.
    # Sort by distance from center, then pick first 12?
    # This will give 1 center, 6 ring 1, 5 ring 2 (the first 5 encountered).
    # The order depends on the loop.
    # Let's sort by distance, then angle to make it deterministic and compact.
    def get_dist_angle(p):
        return (np.sqrt(p[0]**2 + p[1]**2), np.arctan2(p[1], p[0]))
        
    pixel_centers.sort(key=get_dist_angle)
    selected_centers = pixel_centers[:12]
    
    # Now find minimal enclosing hexagon
    # Try rotations
    best_side = float('inf')
    best_angle = 0
    
    # The container hexagon has side S.
    # Normals are at angles alpha, alpha+60, ...
    # Constraint: |p . n| <= S * sqrt(3) / 2
    
    for deg in range(0, 60, 1):
        alpha = np.deg2rad(deg)
        max_proj = 0
        for k in range(6):
            n_angle = alpha + np.deg2rad(60 * k)
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            for x, y in selected_centers:
                proj = abs(x * nx + y * ny)
                if proj > max_proj:
                    max_proj = proj
        # S * sqrt(3) / 2 >= max_proj
        # S >= max_proj * 2 / sqrt(3)
        side = max_proj * 2 / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = deg
            
    # Also, we might need to shift the center of the container?
    # The problem allows the outer hexagon to have any center.
    # Usually, centering on the centroid of the inner hexagons is optimal for symmetric shapes.
    # Let's compute the centroid of selected_centers and shift them to 0,0 before calculating.
    
    cx = sum(p[0] for p in selected_centers) / 12
    cy = sum(p[1] for p in selected_centers) / 12
    shifted_centers = [(x - cx, y - cy) for x, y in selected_centers]
    
    best_side = float('inf')
    best_angle = 0
    for deg in range(0, 60, 1):
        alpha = np.deg2rad(deg)
        max_proj = 0
        for k in range(6):
            n_angle = alpha + np.deg2rad(60 * k)
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            for x, y in shifted_centers:
                proj = abs(x * nx + y * ny)
                if proj > max_proj:
                    max_proj = proj
        side = max_proj * 2 / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = deg
            
    # Return inner hexagons. 
    # The inner hexagons have side 1 and angle 0.
    # We need to return (x, y, angle_degrees).
    # The outer hexagon is centered at (cx, cy) with side best_side and angle best_angle.
    
    inner = [[x, y, 0.0] for x, y in selected_centers]
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
def construct():
    # Re-implementation of the logic above inside the block
    import numpy as np
    
    # Generate 19 hexes in a k=3 cluster
    hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= 2:
                hexes.append((q, r))
                
    pixel_centers = []
    for q, r in hexes:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pixel_centers.append((x, y))
        
    def get_dist_angle(p):
        return (np.sqrt(p[0]**2 + p[1]**2), np.arctan2(p[1], p[0]))
        
    pixel_centers.sort(key=get_dist_angle)
    selected_centers = pixel_centers[:12]
    
    cx = sum(p[0] for p in selected_centers) / 12
    cy = sum(p[1] for p in selected_centers) / 12
    shifted_centers = [(x - cx, y - cy) for x, y in selected_centers]
    
    best_side = float('inf')
    best_angle = 0
    # Fine search for angle
    for deg in np.arange(0, 60, 0.5):
        alpha = np.deg2rad(deg)
        max_proj = 0
        for k in range(6):
            n_angle = alpha + np.deg2rad(60 * k)
            nx = np.cos(n_angle)
            ny = np.sin(n_angle)
            for x, y in shifted_centers:
                proj = abs(x * nx + y * ny)
                if proj > max_proj:
                    max_proj = proj
        side = max_proj * 2 / np.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = deg
            
    inner = [[x, y, 0.0] for x, y in selected_centers]
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
