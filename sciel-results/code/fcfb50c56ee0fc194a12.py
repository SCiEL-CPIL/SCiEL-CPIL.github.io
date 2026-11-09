"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # We have 19 centers, need to choose 12
    # The outer hexagon side length L must satisfy:
    # For each inner hex center (cx, cy), the hexagon's vertices must be inside outer hex
    
    # Let's use flat-top: vertices at (L*cos(60k), L*sin(60k)) for k=0..5
    # For a regular hexagon with circumradius R (distance center to vertex = R):
    # Side length s = R (for regular hexagon, side = circumradius)
    # Inradius = L*sqrt3/2
    
    def required_L(centers_subset):
        """Find minimum L such that all unit hexagons at given centers fit inside outer hex of side L."""
        # For each inner hex center, check all 6 vertices
        # Unit hexagon vertices (flat-top, side 1): at angles 0,60,120,180,240,300, distance 1
        min_L = 0.0
        for (cx, cy, _, _) in centers_subset:
            for k in range(6):
                angle = math.radians(60 * k)
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                # Find minimum L such that (vx,vy) is in hex of side L
                # Outer hex is flat-top (normals at 30,90,150,210,270,330)
                # Condition: proj onto normal <= inradius = L * sqrt(3) / 2
                max_proj = 0.0
                for j in range(6):
                    ang = math.radians(60 * j + 30)
                    proj = vx * math.cos(ang) + vy * math.sin(ang)
                    if proj > max_proj:
                        max_proj = proj
                L_needed = max_proj / (math.sqrt(3.0) / 2.0)
                if L_needed > min_L:
                    min_L = L_needed
        return min_L
    
    # Try all subsets of size 12 from 19 centers
    # C(19,12) = C(19,7) = 50388 - manageable
    from itertools import combinations
    
    best_L = float('inf')
    best_subset = None
    
    # Scaling factor to ensure strict non-overlap (gap > 0)
    # The grid centers are at distance sqrt(3) apart.
    # Scaling by 1.000001 creates a small gap.
    SCALE = 1.000001
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        # Scale centers for this subset to check L with gap
        scaled_centers_subset = []
        for i in subset:
            x, y, _, _ = all_centers[i]
            scaled_centers_subset.append((x * SCALE, y * SCALE, 0, 0))
        
        L = required_L(scaled_centers_subset)
        if L < best_L:
            best_L = L
            best_subset = subset
    
    if best_subset is None:
        # Fallback: use first 12
        best_subset = tuple(range(12))
        scaled_centers_subset = []
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            scaled_centers_subset.append((x * SCALE, y * SCALE, 0, 0))
        best_L = required_L(scaled_centers_subset)
    
    # Construct inner hexagons with scaled coordinates
    inner = []
    for i in best_subset:
        x, y, _, _ = all_centers[i]
        inner.append([x * SCALE, y * SCALE, 0.0])
    
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
