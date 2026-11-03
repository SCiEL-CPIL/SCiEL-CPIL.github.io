import numpy as np
from scipy.spatial import ConvexHull
import time
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal lattice: use axial coordinates (q, r)
    # Convert axial to Cartesian: x = sqrt(3)*(q + r/2), y = 3*r/2  (pointy-top)
    # Or for flat-top: x = 3/2*q, y = sqrt(3)*(r + q/2)
    # Let's use pointy-top: distance between adjacent centers = sqrt(3)
    
    def axial_to_cart(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 3.0 * r / 2.0
        return x, y
    
    # Generate candidate positions in axial coordinates
    # For a hex block of radius R, positions satisfy max(|q|,|r|,|q+r|) <= R
    # We want ~11 positions
    
    # Strategy: try all subsets of size 11 from a candidate pool,
    # or better: try known compact configurations
    
    # Known good configs for N=11:
    # Config 1: Full hex of radius 1 (7 hexes) + 4 more
    # Let's enumerate hex positions in radius 2 (19 positions), pick best 11
    
    # All axial coords with max(|q|,|r|,|q+r|) <= 2
    all_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) <= 2:
                all_positions.append((q, r))
    
    # We need to choose 11 of these 19 positions
    # The enclosing hexagon side length depends on the bounding box in hex space
    
    # For a set of axial coords, the minimal enclosing regular hexagon (aligned with lattice)
    # has side length related to the range of q, r, and q+r
    
    # Actually, let's compute the minimal enclosing hexagon numerically
    
    def min_enclosing_hex_side(points_2d):
        """Find minimal side length of a regular hexagon (any rotation) enclosing all points."""
        # For each angle, project points onto 6 axes, compute required side
        best_side = float('inf')
        best_angle = 0.0
        best_center = (0, 0)
        
        # Try many angles
        for angle_deg in range(0, 180, 1):
            angle = math.radians(angle_deg)
            # 6 normal directions for hexagon faces
            normals = []
            for k in range(6):
                a = angle + k * math.pi / 3
                normals.append((math.cos(a), math.sin(a)))
            
            # For each point, project onto each normal
            # The hexagon is defined by: for each normal n_i, min dot <= dot(p, n_i) <= max dot + side*sqrt(3)
            # Actually, for a regular hexagon with side s, the width in direction n is 2*s*cos(pi/6) = s*sqrt(3)
            # Wait: the distance between opposite faces is 2 * apothem = 2 * s * sqrt(3)/2 = s*sqrt(3)
            
            # So for each normal direction, the range of projections must be <= s * sqrt(3)
            # And the center is at the midpoint of the ranges
            
            min_projs = [float('inf')] * 6
            max_projs = [float('-inf')] * 6
            
            for px, py in points_2d:
                for k in range(6):
                    proj = px * normals[k][0] + py * normals[k][1]
                    min_projs[k] = min(min_projs[k], proj)
                    max_projs[k] = max(max_projs[k], proj)
            
            # Required side: for each direction, range / sqrt(3)
            side = max((max_projs[k] - min_projs[k]) / math.sqrt(3) for k in range(6))
            
            if side < best_side:
                best_side = side
                best_angle = angle
                # Center is average of midpoints
                cx = 0.0
                cy = 0.0
                for k in range(6):
                    cx += (min_projs[k] + max_projs[k]) / 2.0 * normals[k][0]
                    cy += (min_projs[k] + max_projs[k]) / 2.0 * normals[k][1]
                cx /= 6.0
                cy /= 6.0
                best_center = (cx, cy)
        
        return best_side, best_angle, best_center
    
    # Try different subsets: use a heuristic - start with center, add nearest neighbors
    # Or just try the full radius-1 hex (7) plus 4 from radius-2 ring
    
    # Positions sorted by distance from origin
    pos_with_dist = []
    for q, r in all_positions:
        x, y = axial_to_cart(q, r)
        dist = math.sqrt(x*x + y*y)
        pos_with_dist.append((dist, q, r, x, y))
    pos_with_dist.sort()
    
    # Greedy: start with closest 11
    best_config = None
    best_side = float('inf')
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: closest 11
    closest_11 = pos_with_dist[:11]
    strategies.append(closest_11)
    
    # Strategy 2: full inner hex (7) + 4 from outer ring, try different choices
    inner_7 = pos_with_dist[:7]  # should be the 7 closest (center + 6 neighbors)
    outer_ring = [p for p in pos_with_dist if p[0] > inner_7[-1][0] + 0.1]
    
    # Try adding 4 from outer ring in different combinations
    import itertools
    for combo in itertools.combinations(range(len(outer_ring)), 4):
        config = inner_7 + [outer_ring[i] for i in combo]
        strategies.append(config)
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        break
    
    # Evaluate each strategy
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        pts = [(x, y) for (_, _, _, x, y) in strategy]
        side, angle, center = min_enclosing_hex_side(pts)
        
        if side < best_side:
            best_side = side
            best_config = (strategy, angle, center)
    
    if best_config is None:
        # Fallback
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0
    
    strategy, angle, center = best_config
    cx, cy = center
    
    # Build inner list: for each hex, position relative to outer center, rotation 0
    inner = []
    for (_, q, r, x, y) in strategy:
        # Position relative to outer hex center
        dx = x - cx
        dy = y - cy
        inner.append([dx, dy, 0.0])
    
    return inner, (cx, cy), best_side, angle
# EVOLVE-BLOCK-END
