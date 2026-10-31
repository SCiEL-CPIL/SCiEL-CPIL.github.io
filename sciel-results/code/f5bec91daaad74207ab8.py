import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the lattice vectors for a flat-top hexagonal grid
    # Distance between centers is sqrt(3)
    # Axial coordinates to Cartesian:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # We want to place 11 hexagons.
    # 1 at (0,0)
    # 6 at distance 1 (k=1 ring)
    # 4 at distance 2 (k=2 ring)
    
    inner = []
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # k=1 ring (6 hexagons)
    # Axial coords: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    k1_coords = [
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1)
    ]
    
    for q, r in k1_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # k=2 ring (need 4 more)
    # We pick 4 corners to ensure symmetry and minimal bounding box
    # Corners are at (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    k2_coords = [
        (2, 0),
        (0, 2),
        (-2, 2),
        (-2, 0)
    ]
    
    for q, r in k2_coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Total 11 hexagons
    
    # Outer hexagon
    # Side length L = 2 * sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1.0
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
