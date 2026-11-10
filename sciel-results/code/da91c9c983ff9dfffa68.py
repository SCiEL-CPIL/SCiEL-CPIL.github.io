import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Centers of the 11 hexagons in Cartesian coordinates
    # Configuration: 7-hex cluster + 4 hexes in a cross pattern (Axial coords used for generation)
    # Axial (q, r) -> Cartesian (x, y) for flat-top hexagons of side 1
    # x = 1.5 * q
    # y = sqrt(3) * (r + q / 2)
    
    sqrt3 = math.sqrt(3)
    
    # Define the 11 axial coordinates
    axial_coords = [
        (0, 0),   # Center
        (1, 0),   # Right
        (0, 1),   # Top
        (-1, 1),  # Top-Left
        (-1, 0),  # Left
        (0, -1),  # Bottom
        (1, -1),  # Bottom-Right
        
        (2, 0),   # Far Right
        (0, 2),   # Far Top
        (-2, 0),  # Far Left
        (0, -2)   # Far Bottom
    ]
    
    inner = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        # All hexagons are aligned with the outer hexagon (rotation 0)
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # Side length 5.0 is sufficient for this configuration.
    # Center at (0,0)
    # Rotation 0.0 (flat-top)
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
