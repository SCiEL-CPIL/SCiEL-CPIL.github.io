"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice points
    # Axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # We need N points. Let's generate a sufficient number and sort by hex distance.
    # Hex distance from origin in axial coords is (|q| + |r| + |q+r|) / 2
    # We want the N closest points to the origin to form a compact cluster.
    
    points = []
    # Search radius in lattice steps. For N=11, radius 2 is enough (1 + 6 + 12 > 11).
    # Let's go up to radius 5 to be safe for larger N if this code is reused/evolved.
    max_radius = 10 
    for r in range(-max_radius, max_radius + 1):
        for q in range(-max_radius, max_radius + 1):
            # Calculate hex distance
            dist = (abs(q) + abs(r) + abs(q + r)) / 2.0
            # Calculate cartesian coordinates
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            points.append((dist, x, y))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[0])
    
    # Take the first N points
    selected = points[:N]
    
    inner = []
    max_proj = 0.0
    
    # Normals for a hexagon with vertices at angles 0, 60, 120...
    # Sides are perpendicular to 30, 90, 150...
    # n1 = (sqrt(3)/2, 1/2)
    # n2 = (0, 1)
    # n3 = (-sqrt(3)/2, 1/2)
    
    sqrt3_2 = np.sqrt(3) / 2.0
    
    for dist, x, y in selected:
        inner.append([x, y, 0.0])
        
        # Calculate max projection on the 3 unique normal directions
        # We take absolute values because the hexagon is symmetric
        proj1 = abs(x * sqrt3_2 + y * 0.5)
        proj2 = abs(y)
        proj3 = abs(-x * sqrt3_2 + y * 0.5)
        
        current_max = max(proj1, proj2, proj3)
        if current_max > max_proj:
            max_proj = current_max
            
    # The condition for containment is:
    # max_proj <= (S - 1) * (sqrt(3) / 2)
    # So S = 1 + max_proj / (sqrt(3) / 2)
    
    side = 1.0 + max_proj / sqrt3_2
    
    # Center is (0,0)
    center = (0.0, 0.0)
    
    # Outer angle 0.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
