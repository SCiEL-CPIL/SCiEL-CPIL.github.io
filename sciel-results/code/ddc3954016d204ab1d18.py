"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place 11 unit hexagons in a compact arrangement
    # Using triangular lattice with spacing sqrt(3), all same orientation
    d = math.sqrt(3)  # center-to-center distance for touching hexagons
    
    # Positions (x, y) for 11 hexagons
    positions = [
        (0, 0),
        (d, 0), (-d, 0),
        (d/2, 1.5), (-d/2, 1.5),
        (d/2, -1.5), (-d/2, -1.5),
        (0, 3),
        (d, 3), (-d, 3),
        (0, -3),
    ]
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Compute minimal outer hexagon side length
    # Outer hexagon centered at origin, angle 0 (flat-top)
    # 6 normal directions (unit vectors)
    normals = [
        (math.sqrt(3)/2, 0.5),
        (0, 1),
        (-math.sqrt(3)/2, 0.5),
        (-math.sqrt(3)/2, -0.5),
        (0, -1),
        (math.sqrt(3)/2, -0.5),
    ]
    
    apothem_inner = math.sqrt(3) / 2  # apothem of unit hexagon
    
    max_extent = 0
    for cx, cy in positions:
        for nx, ny in normals:
            extent = cx * nx + cy * ny + apothem_inner
            if extent > max_extent:
                max_extent = extent
    
    # Outer apothem must be >= max_extent
    # Outer apothem = S * sqrt(3) / 2
    S = (2 / math.sqrt(3)) * max_extent
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
