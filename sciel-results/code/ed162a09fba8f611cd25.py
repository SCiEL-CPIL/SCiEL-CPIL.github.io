"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagonal lattice cluster
    # Axial coordinates (q, r) with hex distance <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex distance from origin
            dist = (abs(q) + abs(r) + abs(q + r)) // 2
            if dist <= 2:
                # Flat-top hexagon lattice
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, dist))
    
    # We have 19 centers. Select 12 that minimize the required outer hexagon side.
    # Strategy: take all dist-0 and dist-1 (7 centers), then pick 5 from dist-2.
    # The dist-2 centers are at the "corners" and "edges". 
    # For a flat-top outer hexagon aligned with the lattice, all 19 fit in L=3.
    # So just take any 12.
    
    # Sort by distance, take closest 12
    all_centers.sort(key=lambda c: c[2])
    selected = all_centers[:N]
    
    inner = [[x, y, 0.0] for x, y, d in selected]
    
    # Outer hexagon: center at origin, side length 3, rotation 0 (flat-top aligned)
    center = (0.0, 0.0)
    side = 3.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
