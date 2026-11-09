import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define helper to check if a point is inside outer hexagon of side S (flat-top, centered at 0,0)
    def is_inside_outer(x, y, S):
        if abs(x) > S: return False
        if abs(y) > S * math.sqrt(3) / 2: return False
        if abs(math.sqrt(3)*x + y) > math.sqrt(3)*S: return False
        if abs(-math.sqrt(3)*x + y) > math.sqrt(3)*S: return False
        return True

    def get_hex_vertices(cx, cy, angle_deg):
        # Flat-top hexagon vertices relative to center
        # Side length 1
        # Angles: 0, 60, 120, 180, 240, 300
        verts = []
        for i in range(6):
            theta = math.radians(60 * i + angle_deg)
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            verts.append((vx, vy))
        return verts

    def min_side_for_config(centers, angles, outer_angle=0.0):
        # We can rotate the whole config to find better fit, 
        # but simpler: just check if current config fits in S.
        # Since outer hex is symmetric, rotating inner config by theta is equivalent 
        # to rotating outer hex by -theta. 
        # We return the min S for the given centers/angles relative to outer at 0.
        
        # Binary search for S
        lo = 0.0
        hi = 100.0
        
        def valid(S):
            for i in range(len(centers)):
                cx, cy = centers[i]
                ang = angles[i]
                verts = get_hex_vertices(cx, cy, ang)
                for vx, vy in verts:
                    if not is_inside_outer(vx, vy, S):
                        return False
            return True
        
        for _ in range(50):
            mid = (lo + hi) / 2
            if valid(mid):
                hi = mid
            else:
                lo = mid
        return hi

    # Generate Config 1: 3x4 Grid (Flat-Top aligned)
    # Cols 0,1,2. Rows 0,1,2,3.
    centers1 = []
    angles1 = []
    for c in range(3):
        for r in range(4):
            x = 1.5 * c
            y = math.sqrt(3) * (r + 0.5 * (c % 2))
            centers1.append((x, y))
            angles1.append(0.0)
    
    # Center the config
    cx_avg = sum(p[0] for p in centers1) / len(centers1)
    cy_avg = sum(p[1] for p in centers1) / len(centers1)
    centers1 = [(x - cx_avg, y - cy_avg) for x, y in centers1]
    
    best_S = min_side_for_config(centers1, angles1)
    best_config = (centers1, angles1, best_S)

    # Generate Config 2: 2x6 Grid
    centers2 = []
    angles2 = []
    for c in range(2):
        for r in range(6):
            x = 1.5 * c
            y = math.sqrt(3) * (r + 0.5 * (c % 2))
            centers2.append((x, y))
            angles2.append(0.0)
            
    cx_avg2 = sum(p[0] for p in centers2) / len(centers2)
    cy_avg2 = sum(p[1] for p in centers2) / len(centers2)
    centers2 = [(x - cx_avg2, y - cy_avg2) for x, y in centers2]
    
    S2 = min_side_for_config(centers2, angles2)
    if S2 < best_S[2]:
        best_config = (centers2, angles2, S2)

    # Generate Config 3: Hexagonal 7 + 5
    # 7 hexes: (0,0) and 6 neighbors
    # Neighbors in flat-top:
    # (1.5, sqrt(3)/2), (1.5, -sqrt(3)/2), (0, sqrt(3)), (-1.5, sqrt(3)/2), (-1.5, -sqrt(3)/2), (0, -sqrt(3))
    # Wait, standard flat-top neighbors:
    # E: (1.5, sqrt(3)/2) ? No.
    # Let's use axial to cartesian for flat-top:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Neighbors of (0,0) in axial (q,r):
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    hex7_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Convert to cartesian
    centers3 = []
    for q, r in hex7_axial:
        x = 1.5 * q
        y = math.sqrt(3) * (r + q/2)
        centers3.append((x, y))
        
    # Add 5 more. Attach to the "outer" side of the hex7 cluster.
    # The hex7 cluster has 6 sides. Each side has 1 hex in the middle (the ones at dist 1).
    # To extend a side, we add a hex adjacent to the side-hex, moving away from center.
    # Side-hex at (1,0). Its outward neighbor is (2,0)? No, (2,0) is dist 2.
    # Let's just pick 5 of the 6 "corners" of the dist-2 ring?
    # Or just add 5 hexes in a line on one side?
    # Let's try adding 5 hexes to the right side.
    # Current right-most hexes are at q=1.
    # Add (2,0), (2,-1), (2,-2)? No, that's a line.
    # Let's try a more symmetric addition: Add 5 hexes to 5 of the 6 directions.
    # Directions from center:
    # 1. (1,0) -> Add (2,0)
    # 2. (1,-1) -> Add (2,-1)
    # 3. (0,-1) -> Add (0,-2)
    # 4. (-1,0) -> Add (-2,0)
    # 5. (-1,1) -> Add (-2,1)
    # Skip (0,1).
    
    additions = [(2,0), (2,-1), (0,-2), (-2,0), (-2,1)]
    for q, r in additions:
        x = 1.5 * q
        y = math.sqrt(3) * (r + q/2)
        centers3.append((x, y))
        
    angles3 = [0.0] * 12
    
    cx_avg3 = sum(p[0] for p in centers3) / len(centers3)
    cy_avg3 = sum(p[1] for p in centers3) / len(centers3)
    centers3 = [(x - cx_avg3, y - cy_avg3) for x, y in centers3]
    
    S3 = min_side_for_config(centers3, angles3)
    if S3 < best_config[2]:
        best_config = (centers3, angles3, S3)

    # Try rotating the best config by 30 degrees?
    # If we rotate the inner hexes by 30 deg, they become pointy-top.
    # This might fit better in a flat-top outer hex?
    # Let's test the best config with angles = 30.
    # Note: If we rotate the hexes, their centers stay the same? No, the shape changes.
    # But the "footprint" might be smaller in some directions.
    
    # Actually, let's just try rotating the entire coordinate system of the best config.
    # This is equivalent to changing the outer angle.
    # Since we return the outer angle, we can just set it to 0 and rotate the inner points.
    
    # Let's refine the search for the best rotation of the best config.
    best_overall_S = best_config[2]
    best_rotation = 0.0
    best_c = best_config[0]
    best_a = best_config[1]
    
    # Try a few rotations
    for rot_deg in range(0, 60, 5):
        rad = math.radians(rot_deg)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        
        rotated_centers = []
        for x, y in best_c:
            nx = x * cos_r - y * sin_r
            ny = x * sin_r + y * cos_r
            rotated_centers.append((nx, ny))
            
        # The hexes themselves are also rotated?
        # If we rotate the whole assembly, the relative angles remain 0.
        # So we just check the new centers against the outer hex.
        S_rot = min_side_for_config(rotated_centers, best_a)
        if S_rot < best_overall_S:
            best_overall_S = S_rot
            best_rotation = rot_deg
            best_c = rotated_centers

    # Format output
    inner = [[x, y, a] for (x, y), a in zip(best_c, best_a)]
    return inner, (0.0, 0.0), best_overall_S, best_rotation
# EVOLVE-BLOCK-END
