import numpy as np
from math import sqrt, cos, sin, pi

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Pack N unit hexagons in a hexagonal lattice
    # Find smallest k such that hexagonal number H_k = 1 + 3k(k+1) >= N
    k = 0
    while 1 + 3 * k * (k + 1) < N:
        k += 1
    
    # Generate axial coordinates for hexagonal cluster of radius k
    centers = []
    for q in range(-k, k + 1):
        r_min = max(-k, -q - k)
        r_max = min(k, -q + k)
        for r in range(r_min, r_max + 1):
            centers.append((q, r))
    
    # Take first N centers
    centers = centers[:N]
    
    # Convert to Cartesian (hexagons with circumradius 1, same orientation)
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 3/2*r
    cartesian = [(sqrt(3) * (q + r / 2.0), 1.5 * r) for q, r in centers]
    
    inner = [[x, y, 0.0] for x, y in cartesian]
    
    # Compute minimal outer hexagon side length
    # Outer hexagon: center (0,0), circumradius = side length S, vertices at angles 0,60,120,180,240,300
    # Edge normals at angles 30,90,150,210,270,330
    # Inradius = S * sqrt(3)/2
    # Constraint: for all inner hex vertices (vx,vy) and all 6 edge normals n_j:
    #   |dot((vx,vy), n_j)| <= S * sqrt(3)/2
    
    max_abs_proj = 0.0
    for x, y, _ in inner:
        for i in range(6):
            angle = i * pi / 3
            vx = x + cos(angle)
            vy = y + sin(angle)
            for j in range(6):
                na = (j * 60 + 30) * pi / 180
                nx = cos(na)
                ny = sin(na)
                proj = abs(vx * nx + vy * ny)
                if proj > max_abs_proj:
                    max_abs_proj = proj
    
    side_length = max_abs_proj / (sqrt(3) / 2)
    
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
