import numpy as np
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 inner hexagon centers in a honeycomb arrangement
    # Using axial coordinates (q, r) to cartesian (x, y) for flat-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # We want 12 hexagons.
    # A central cluster of 7 is standard.
    # Add 5 more to the second ring.
    
    # Axial coordinates for the 7 central hexagons:
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    
    # Axial coordinates for the 5 additional hexagons from the second ring:
    # We pick 5 from the 12 available in ring 2 to keep it compact.
    # Ring 2 coords: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,1), (0,2), (1,2)
    # Note: (-1,1) and (0,1) and (1,0) etc are ring 1.
    # Let's list valid ring 2 again carefully.
    # Distance from (0,0) in hex grid is (|q| + |r| + |q+r|) / 2
    # (2,0) -> (2+0+2)/2 = 2
    # (2,-1) -> (2+1+1)/2 = 2
    # (2,-2) -> (2+2+0)/2 = 2
    # (1,-2) -> (1+2+1)/2 = 2
    # (0,-2) -> (0+2+2)/2 = 2
    # (-1,-2) -> (1+2+3)/2 = 3? No. |-1|+|-2|+|-3| = 6/2=3. Wait.
    # Let's use cube coordinates to be safe or just verify distances.
    # Hex distance between (0,0) and (q,r) is (|q| + |r| + |q+r|) / 2.
    
    # Ring 1:
    # (1,0): (1+0+1)/2 = 1
    # (1,-1): (1+1+0)/2 = 1
    # (0,-1): (0+1+1)/2 = 1
    # (-1,0): (1+0+1)/2 = 1
    # (-1,1): (1+1+0)/2 = 1
    # (0,1): (0+1+1)/2 = 1
    
    # Ring 2 candidates:
    # (2,0): (2+0+2)/2 = 2
    # (2,-1): (2+1+1)/2 = 2
    # (2,-2): (2+2+0)/2 = 2
    # (1,-2): (1+2+1)/2 = 2
    # (0,-2): (0+2+2)/2 = 2
    # (-1,-2): (1+2+3)/2 = 3 -> NOT RING 2
    # (-2,-1): (2+1+3)/2 = 3 -> NOT RING 2
    # (-2,0): (2+0+2)/2 = 2
    # (-2,1): (2+1+1)/2 = 2
    # (-1,1): RING 1
    # (0,2): (0+2+2)/2 = 2
    # (1,2): (1+2+3)/2 = 3 -> NOT RING 2
    
    # So Ring 2 points are:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-2,0), (-2,1), (0,2)
    # Wait, I missed some?
    # Let's iterate all (q,r) with small values and check dist.
    points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            d = (abs(q) + abs(r) + abs(q+r)) // 2
            if d == 0:
                points.append((q,r))
            elif d == 1:
                points.append((q,r))
            elif d == 2:
                # We only need 5 more
                pass
    
    # We have 7 points from d=0 and d=1.
    # We need 5 from d=2.
    # Let's collect d=2 points.
    ring2 = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            d = (abs(q) + abs(r) + abs(q+r)) // 2
            if d == 2:
                ring2.append((q,r))
                
    # Select 5 points from ring2.
    # To minimize the bounding box, we should pick points that are symmetric or fill the shape.
    # A common compact shape for 12 is a "hexagon with a bite" or just random 5.
    # Actually, the optimal packing might be slightly rotated or different.
    # But let's stick to the grid for now.
    # Let's pick 5 that are spread out to avoid clumping, or clumped to reduce max radius?
    # Clumping reduces the max radius.
    # If we pick 5 adjacent ones, the shape is less round, but the max radius might be smaller?
    # No, a rounder shape usually fits in a smaller circle/hexagon.
    # Let's pick 5 that are roughly evenly distributed?
    # Or just the first 5 found?
    # Let's try to pick them to minimize the max distance from origin.
    # All d=2 points have the same hex-distance, but different Euclidean distances.
    # Euclidean distance squared:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # dist2 = 3*(q+r/2)^2 + 2.25*r^2
    
    # Calculate dist2 for each ring2 point
    scored_ring2 = []
    for q, r in ring2:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        d2 = x*x + y*y
        scored_ring2.append((d2, q, r))
        
    # Sort by distance? No, we want to pick 5.
    # If we pick the 5 closest to the origin, we get a smaller cluster.
    scored_ring2.sort()
    selected_ring2 = [ (q,r) for _, q, r in scored_ring2[:5] ]
    
    all_axial = points + selected_ring2
    
    # Convert to cartesian
    centers = []
    for q, r in all_axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Calculate minimal outer side length S
    # The outer hexagon is flat-top, centered at (0,0).
    # A point (px, py) is inside a flat-top hexagon of side S if:
    # 1. |px| <= S
    # 2. |py| <= S * sqrt(3) / 2
    # 3. |py| <= sqrt(3) * (S - |px|)  (This is the slanted side constraint)
    # Actually, the vertices of the outer hexagon are at (S, 0), (S/2, S*sqrt(3)/2), etc.
    # The lines are:
    # x = S
    # x = -S
    # y = S*sqrt(3)/2
    # y = -S*sqrt(3)/2
    # y = sqrt(3) * (S - x)  => sqrt(3)x + y = sqrt(3)S
    # y = sqrt(3) * (S + x)  => -sqrt(3)x + y = sqrt(3)S
    # y = -sqrt(3) * (S - x) => sqrt(3)x - y = sqrt(3)S
    # y = -sqrt(3) * (S + x) => -sqrt(3)x - y = sqrt(3)S
    
    # For each inner hexagon center (cx, cy), we need to ensure all 6 vertices are inside.
    # The vertices of an inner hexagon at (cx, cy) are:
    # (cx + 1, cy)
    # (cx + 0.5, cy + sqrt(3)/2)
    # (cx - 0.5, cy + sqrt(3)/2)
    # (cx - 1, cy)
    # (cx - 0.5, cy - sqrt(3)/2)
    # (cx + 0.5, cy - sqrt(3)/2)
    
    # Instead of checking all vertices for all hexagons, we can find the max "required S" for each hexagon.
    # For a point (px, py), the required S is the max of the 6 constraints solved for S.
    # From x = S -> S >= px
    # From x = -S -> S >= -px
    # From y = S*sqrt(3)/2 -> S >= 2*py/sqrt(3)
    # From y = -S*sqrt(3)/2 -> S >= -2*py/sqrt(3)
    # From sqrt(3)x + y = sqrt(3)S -> S >= x + y/sqrt(3)
    # From -sqrt(3)x + y = sqrt(3)S -> S >= -x + y/sqrt(3)
    # From sqrt(3)x - y = sqrt(3)S -> S >= x - y/sqrt(3)
    # From -sqrt(3)x - y = sqrt(3)S -> S >= -x - y/sqrt(3)
    
    # So S_point = max(px, -px, 2*py/sqrt(3), -2*py/sqrt(3), x+y/sqrt(3), -x+y/sqrt(3), x-y/sqrt(3), -x-y/sqrt(3))
    # Note: Some of these are redundant.
    # max(px, -px) = |px|
    # max(2*py/sqrt(3), -2*py/sqrt(3)) = 2*|py|/sqrt(3)
    # The slanted ones:
    # max(x+y/sqrt(3), -x+y/sqrt(3), x-y/sqrt(3), -x-y/sqrt(3))
    # = max(|x| + |y|/sqrt(3) ? No.
    # Let's just compute the max of the 8 values.
    
    max_S = 0.0
    sqrt3 = math.sqrt(3)
    
    for cx, cy in centers:
        # Get vertices
        for i in range(6):
            angle = math.radians(60 * i)
            px = cx + math.cos(angle)
            py = cy + math.sin(angle)
            
            # Calculate required S for this point
            s1 = abs(px)
            s2 = abs(2 * py / sqrt3)
            s3 = abs(px + py / sqrt3)
            s4 = abs(-px + py / sqrt3)
            # Note: s3 and s4 cover the slanted sides?
            # Line 1: sqrt(3)x + y = sqrt(3)S  => S = x + y/sqrt(3)
            # Line 2: -sqrt(3)x + y = sqrt(3)S => S = -x + y/sqrt(3)
            # Line 3: sqrt(3)x - y = sqrt(3)S => S = x - y/sqrt(3)
            # Line 4: -sqrt(3)x - y = sqrt(3)S => S = -x - y/sqrt(3)
            
            # We need S >= all these values.
            # Since S is positive, we take the max of the absolute values of the RHS?
            # Actually, the constraints are inequalities of the form A*x + B*y <= C*S.
            # So S >= (A*x + B*y)/C.
            # We need S >= max( (A*x + B*y)/C ) for all sides.
            
            # Side 1 (Right): x <= S => S >= x
            # Side 2 (Left): -x <= S => S >= -x
            # Side 3 (Top): y <= S*sqrt(3)/2 => S >= 2*y/sqrt(3)
            # Side 4 (Bottom): -y <= S*sqrt(3)/2 => S >= -2*y/sqrt(3)
            # Side 5 (Top-Right): sqrt(3)x + y <= S*sqrt(3) => S >= x + y/sqrt(3)
            # Side 6 (Top-Left): -sqrt(3)x + y <= S*sqrt(3) => S >= -x + y/sqrt(3)
            # Side 7 (Bottom-Right): sqrt(3)x - y <= S*sqrt(3) => S >= x - y/sqrt(3)
            # Side 8 (Bottom-Left): -sqrt(3)x - y <= S*sqrt(3) => S >= -x - y/sqrt(3)
            
            req_s = max(px, -px, 2*py/sqrt3, -2*py/sqrt3, 
                        px + py/sqrt3, -px + py/sqrt3, 
                        px - py/sqrt3, -px - py/sqrt3)
            
            if req_s > max_S:
                max_S = req_s
                
    # Add a small epsilon to ensure strict containment if needed, or just return max_S
    # The problem says "disjoint", so touching is allowed? "disjoint" usually means interiors don't overlap.
    # If they touch, it's valid.
    
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    return inner, (0.0, 0.0), max_S, 0.0
# EVOLVE-BLOCK-END
