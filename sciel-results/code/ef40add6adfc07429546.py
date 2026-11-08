"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define positions for 11 hexagons in a compact 3-4-4 grid arrangement
    # This arrangement minimizes the bounding box compared to a linear one.
    # Row 0: 4 hexes, Row 1: 3 hexes, Row 2: 4 hexes
    # Staggered to fit in hexagonal lattice gaps.
    
    hexes = []
    
    # Row 0 (y = -1.5)
    y0 = -1.5
    for i in range(4):
        # Centered x positions: -1.5, -0.5, 0.5, 1.5 (scaled by sqrt(3))
        x = (i - 1.5) * math.sqrt(3)
        hexes.append([x, y0, 0.0])
        
    # Row 1 (y = 0.0)
    y1 = 0.0
    for i in range(3):
        # Staggered x positions: -1, 0, 1 (scaled by sqrt(3))
        x = (i - 1) * math.sqrt(3)
        hexes.append([x, y1, 0.0])
        
    # Row 2 (y = 1.5)
    y2 = 1.5
    for i in range(4):
        # Centered x positions: -1.5, -0.5, 0.5, 1.5 (scaled by sqrt(3))
        x = (i - 1.5) * math.sqrt(3)
        hexes.append([x, y2, 0.0])
        
    # The cluster is centered at (0,0)
    # Max distance from center to a vertex of the inner hexagons is approx 4.0
    # We use a side length of 4.5 for the outer hexagon to ensure containment
    # with a small margin.
    outer_side = 4.5
    
    return hexes, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
