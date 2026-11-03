import math

N = 12

def construct():
    # Lattice basis vectors for unit hexagons
    # Distance between centers is sqrt(3)
    a = (math.sqrt(3), 0.0)
    b = (math.sqrt(3) / 2.0, 1.5)
    
    inner = []
    
    # 7 central hexagons (Radius 1 in lattice steps)
    # (0,0)
    inner.append([0.0, 0.0, 0.0])
    
    # 6 neighbors
    neighbors = [
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    for i, j in neighbors:
        x = i * a[0] + j * b[0]
        y = i * a[1] + j * b[1]
        inner.append([x, y, 0.0])
        
    # We need 5 more.
    # Next closest lattice points are at distance 3 (lattice step distance squared = 3*3=9? No, dist=3)
    # Lattice coords for distance 3:
    # (1,1), (1,-2), (-2,1), (-1,-1), (-1,2), (2,-1)
    # We pick 5 of them.
    extra_candidates = [
        (1, 1),
        (1, -2),
        (-2, 1),
        (-1, -1),
        (-1, 2)
    ]
    
    for i, j in extra_candidates:
        x = i * a[0] + j * b[0]
        y = i * a[1] + j * b[1]
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Angle 30.0 degrees to align vertices with the extra hexagon centers
    center = (0.0, 0.0)
    side = 4.0
    angle = 30.0
    
    return inner, center, side, angle
