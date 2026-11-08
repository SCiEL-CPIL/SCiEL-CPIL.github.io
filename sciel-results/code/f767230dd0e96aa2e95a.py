import numpy as np
import math
import time

N = 11

def axial_to_cartesian(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def get_min_hexagon_side(points, angle_deg=0):
    """
    Calculate the minimum side length S of a regular hexagon (flat-top, center 0,0)
    that contains all points in 'points', after rotating 'points' by angle_deg.
    """
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    
    max_val = 0.0
    for x, y in points:
        # Rotate point
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        
        # Constraints for flat-top hexagon of side S centered at 0:
        # 1. |ry| <= S * sqrt(3) / 2
        # 2. |sqrt(3) * rx + ry| <= S * sqrt(3)
        # 3. |sqrt(3) * rx - ry| <= S * sqrt(3)
        
        # From 1: S >= 2 * |ry| / sqrt(3)
        s1 = 2.0 * abs(ry) / math.sqrt(3)
        
        # From 2: S >= |sqrt(3) * rx + ry| / sqrt(3)
        s2 = abs(math.sqrt(3) * rx + ry) / math.sqrt(3)
        
        # From 3: S >= |sqrt(3) * rx - ry| / sqrt(3)
        s3 = abs(math.sqrt(3) * rx - ry) / math.sqrt(3)
        
        max_val = max(max_val, s1, s2, s3)
        
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Generate candidate configurations in axial coordinates
    # We will use a list of (q, r) tuples
    
    candidates = []
    
    # Config 1: Cross (1 center, 6 neighbors, 4 at dist 2 along axes)
    c1 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (2,0), (0,2), (-2,0), (0,-2)]
    candidates.append(c1)
    
    # Config 2: 3x3 block (9 hexes) + 2
    # 3x3 block in axial: q in [-1,1], r in [-1,1] such that |q+r|<=1? No, that's a rhombus.
    # A hexagon of side 2 has 7 hexes.
    # Let's take the 7 hexes of side 2, and add 2 from the next shell.
    # Next shell has 12. Add 2 adjacent ones.
    base7 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Add (2,0) and (2,-1)?
    c2 = base7 + [(2,0), (2,-1)]
    candidates.append(c2)
    
    # Config 3: Linear-ish
    # 5 in a row, 3 above, 3 below?
    # Let's just use a grid search for the best subset of a small lattice.
    # Lattice points within radius 3
    lattice_points = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            # Hex distance
            dist = (abs(q) + abs(r) + abs(q+r)) / 2
            if dist <= 3:
                lattice_points.append((q,r))
                
    # We want to pick 11 points that minimize the bounding hexagon.
    # This is a combinatorial problem.
    # Given the small N, we can try some heuristic patterns.
    
    # Pattern: "Pentagon"
    # Center, 5 neighbors, 5 more?
    
    # Let's evaluate the candidates found so far.
    best_score = float('inf')
    best_config = None
    best_angle = 0
    
    for config in candidates:
        # Convert to cartesian
        points = [axial_to_cartesian(q, r) for q, r in config]
        
        # Optimize rotation
        # The optimal rotation is likely near 0, 15, 30, 45, 60.
        # Let's do a fine search.
        min_s_for_config = float('inf')
        min_angle = 0
        for i in range(0, 61, 1): # 1 degree steps
            s = get_min_hexagon_side(points, i)
            if s < min_s_for_config:
                min_s_for_config = s
                min_angle = i
        # Refine around best angle
        for i in range(min_angle - 5, min_angle + 6, 1):
             if 0 <= i < 60:
                s = get_min_hexagon_side(points, i)
                if s < min_s_for_config:
                    min_s_for_config = s
                    min_angle = i
        # Even finer
        for i in range(int(min_angle*10)-10, int(min_angle*10)+11, 1):
             angle = i / 10.0
             if 0 <= angle < 60:
                s = get_min_hexagon_side(points, angle)
                if s < min_s_for_config:
                    min_s_for_config = s
                    min_angle = angle

        if min_s_for_config < best_score:
            best_score = min_s_for_config
            best_config = config
            best_angle = min_angle

    # Check time
    if time.time() > deadline:
        # Return best found so far
        pass
    else:
        # Try more configurations if time permits
        # We can try random subsets of the lattice
        import random
        random.seed(42)
        
        # Generate more diverse candidates
        # e.g. shift the center of the cluster
        # Or just try all subsets of size 11 from a small set?
        # The lattice points within radius 2 are 19 points.
        # C(19, 11) is ~ 75582. That's small enough to brute force!
        
        lattice_r2 = [(q,r) for q in range(-2, 3) for r in range(-2, 3) if (abs(q)+abs(r)+abs(q+r))/2 <= 2]
        # 19 points.
        # We need to pick 11.
        # However, the "shape" matters.
        # Let's iterate through all combinations of 11 points from the 19.
        
        from itertools import combinations
        
        # To speed up, we can precompute cartesian coords
        cart_coords = {p: axial_to_cartesian(*p) for p in lattice_r2}
        
        # We will check all combinations.
        # 75k combinations * 60 rotations * 11 points * 3 ops = ~ 1.5M ops. Fast.
        
        for comb in combinations(lattice_r2, 11):
            # Quick check: bounding box in axial space?
            # Just compute min S.
            points = [cart_coords[p] for p in comb]
            
            # Optimize rotation
            min_s = float('inf')
            min_a = 0
            # Coarse search
            for i in range(0, 60, 5):
                s = get_min_hexagon_side(points, i)
                if s < min_s:
                    min_s = s
                    min_a = i
            # Fine search
            for i in range(min_a - 5, min_a + 6, 1):
                if 0 <= i < 60:
                    s = get_min_hexagon_side(points, i)
                    if s < min_s:
                        min_s = s
                        min_a = i
            # Very fine
            for i in range(int(min_a*10)-10, int(min_a*10)+11, 1):
                 angle = i / 10.0
                 if 0 <= angle < 60:
                    s = get_min_hexagon_side(points, angle)
                    if s < min_s:
                        min_s = s
                        min_a = angle

            if min_s < best_score:
                best_score = min_s
                best_config = comb
                best_angle = min_a
                
            if time.time() > deadline - 10:
                break

    # Construct the final result
    # best_config is a tuple of (q,r)
    # We need to return inner rows (x, y, angle_degrees)
    # The inner hexagons can be rotated.
    # In the lattice packing, all hexagons are aligned (angle 0).
    # So angle is 0.
    
    inner = []
    for q, r in best_config:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
        
    # The outer hexagon center is (0,0) in our calculation.
    # But we can shift the whole configuration to center the bounding box?
    # The minimal enclosing hexagon algorithm assumed center (0,0).
    # If we shift the points, the optimal center might change.
    # However, for a symmetric configuration, (0,0) is optimal.
    # For asymmetric ones, we should find the optimal center.
    
    # Let's refine the center and S for the best_config.
    # We have best_config (axial) and best_angle.
    # Rotate points by best_angle.
    points_rot = []
    for q, r in best_config:
        x, y = axial_to_cartesian(q, r)
        rad = math.radians(best_angle)
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        points_rot.append((rx, ry))
        
    # Find optimal center (cx, cy) and S
    # This is a linear programming problem or can be solved by checking vertices.
    # For a hexagon, the optimal center is the intersection of the "midlines" of the bounding strips.
    # Simpler: Grid search for center?
    # Or just use (0,0) if the config is symmetric.
    # Let's do a small local search for center.
    
    min_s_final = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    
    # Initial guess: centroid of points
    cx = sum(p[0] for p in points_rot) / len(points_rot)
    cy = sum(p[1] for p in points_rot) / len(points_rot)
    
    # Local search
    for _ in range(20):
        improved = False
        for dx in [-0.1, 0, 0.1]:
            for dy in [-0.1, 0, 0.1]:
                ncx = cx + dx
                ncy = cy + dy
                max_val = 0.0
                for x, y in points_rot:
                    px = x - ncx
                    py = y - ncy
                    s1 = 2.0 * abs(py) / math.sqrt(3)
                    s2 = abs(math.sqrt(3) * px + py) / math.sqrt(3)
                    s3 = abs(math.sqrt(3) * px - py) / math.sqrt(3)
                    max_val = max(max_val, s1, s2, s3)
                if max_val < min_s_final:
                    min_s_final = max_val
                    best_cx = ncx
                    best_cy = ncy
                    improved = True
        if not improved:
            break
            
    # Update inner coordinates to be relative to best_cx, best_cy?
    # No, the interface asks for absolute coordinates.
    # The outer hexagon is at (best_cx, best_cy).
    # The inner hexagons are at their absolute positions.
    # But we rotated them.
    # So we need to return the rotated coordinates.
    
    final_inner = []
    for q, r in best_config:
        x, y = axial_to_cartesian(q, r)
        rad = math.radians(best_angle)
        rx = x * math.cos(rad) - y * math.sin(rad)
        ry = x * math.sin(rad) + y * math.cos(rad)
        final_inner.append([rx, ry, 0.0])
        
    return final_inner, (best_cx, best_cy), min_s_final, 0.0

# EVOLVE-BLOCK-END
