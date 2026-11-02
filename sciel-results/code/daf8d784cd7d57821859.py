import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon centers in Cartesian coordinates
    # Based on axial coordinates (q, r)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    coords = [
        (0, 0),      # 1
        (1, 0),      # 2
        (-1, 0),     # 3
        (0, 1),      # 4
        (0, -1),     # 5
        (1, -1),     # 6
        (-1, 1),     # 7
        (2, -1),     # 8
        (2, 0),      # 9
        (-2, 1),     # 10
        (-2, 0)      # 11
    ]
    
    inner = []
    sqrt3 = math.sqrt(3)
    for q, r in coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 for all inner hexagons
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    # Center (0,0)
    # Side length calculated to enclose the cluster with 30 degree rotation
    # S = 2 * sqrt(3)
    side = 2.0 * sqrt3
    # Rotation 30 degrees
    angle = 30.0
    
    return inner, (0.0, 0.0), side, angle
# EVOLVE-BLOCK-END
