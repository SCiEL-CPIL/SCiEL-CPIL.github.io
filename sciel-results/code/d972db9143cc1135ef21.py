"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the positions of the 11 unit hexagons
    # Using a "block" arrangement:
    # Row 0 (y=0): 4 hexagons
    # Row 1 (y=1.5): 3 hexagons
    # Row -1 (y=-1.5): 3 hexagons
    # Row 2 (y=3.0): 1 hexagon
    
    # Hexagon geometry (Pointy-Top orientation, side=1)
    # Horizontal distance between centers: sqrt(3)
    # Vertical distance between rows: 1.5
    # Horizontal offset between rows: sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    
    inner = []
    
    # Row 0: 4 hexagons centered at y=0
    # x positions: -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    x_offsets_row0 = [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]
    for x in x_offsets_row0:
        inner.append([x, 0.0, 0.0])
        
    # Row 1: 3 hexagons centered at y=1.5
    # x positions: -sqrt3, 0, sqrt3
    x_offsets_row1 = [-sqrt3, 0.0, sqrt3]
    for x in x_offsets_row1:
        inner.append([x, 1.5, 0.0])
        
    # Row -1: 3 hexagons centered at y=-1.5
    # x positions: -sqrt3, 0, sqrt3
    for x in x_offsets_row1:
        inner.append([x, -1.5, 0.0])
        
    # Row 2: 1 hexagon centered at y=3.0
    inner.append([0.0, 3.0, 0.0])
    
    # Calculate bounding box to center the cluster
    min_x = min(p[0] for p in inner) - sqrt3/2
    max_x = max(p[0] for p in inner) + sqrt3/2
    min_y = min(p[1] for p in inner) - 1.0
    max_y = max(p[1] for p in inner) + 1.0
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Shift all inner hexagons to center the cluster at (0,0)
    for p in inner:
        p[0] -= center_x
        p[1] -= center_y
        
    # The outer hexagon side length
    # Based on analysis, S=4.0 is sufficient for this packing
    # We use a slightly larger value to ensure validity due to floating point precision
    side = 4.01
    
    # The outer hexagon is centered at (0,0) and has angle 0
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
