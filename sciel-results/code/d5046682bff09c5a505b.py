"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Create a 3x4 grid of hexagons
    # Flat-top hexagons
    # Side length 1
    # Width sqrt(3), Height 2
    # Row spacing 1.5
    # Col spacing sqrt(3)
    
    sqrt3 = math.sqrt(3)
    inner = []
    
    # 3 rows, 4 cols
    # Center the block at (0,0)
    # Rows at y = -1.5, 0, 1.5
    # Cols depend on row offset
    
    row_y = [-1.5, 0.0, 1.5]
    
    for i, y in enumerate(row_y):
        # Offset x for odd rows (if we consider 0 as even)
        # Let's say row 0 (y=-1.5) has offset 0
        # Row 1 (y=0) has offset sqrt3/2
        # Row 2 (y=1.5) has offset 0
        # This creates a parallelogram.
        # To make it more symmetric, let's center the x coordinates.
        
        if i == 0 or i == 2:
            # Even rows: x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
            xs = [-1.5 * sqrt3, -0.5 * sqrt3, 0.5 * sqrt3, 1.5 * sqrt3]
        else:
            # Odd row: x = -sqrt3, 0, sqrt3, 2*sqrt3
            # Shift to center?
            # The block is skewed.
            # Let's shift the odd row by -0.5*sqrt3 to center it relative to the others?
            # No, standard tiling.
            xs = [-1.0 * sqrt3, 0.0, 1.0 * sqrt3, 2.0 * sqrt3]
            
        # Center the entire set of points
        # We'll calculate the bounding box and shift later, or just shift now.
        # Let's assume the center of the block is roughly (0.25*sqrt3, 0)
        # Shift x by -0.25*sqrt3 to center.
        shift_x = -0.25 * sqrt3
        
        for x in xs:
            inner.append([x + shift_x, y, 0.0])
            
    # Calculate required side length S
    # Bounding box of the shifted points
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    # Add hexagon extent
    # Hexagon extends by sqrt3/2 in x and 1 in y from center
    # So bounding box of the union:
    # X range: [min_x - sqrt3/2, max_x + sqrt3/2]
    # Y range: [min_y - 1, max_y + 1]
    
    w = (max_x - min_x) + sqrt3
    h = (max_y - min_y) + 2
    
    # Minimal S for flat-top hexagon containing w x h box centered at origin
    # Condition: S >= x + y/sqrt3 for corner (w/2, h/2)
    # Also S >= w/sqrt3 and S >= h/2
    s1 = w / sqrt3
    s2 = h / 2
    s3 = (w / 2) + (h / 2) / sqrt3
    
    S = max(s1, s2, s3)
    
    # Add a small margin for safety
    S = S + 0.1
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
