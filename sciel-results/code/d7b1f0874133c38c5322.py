import math

def construct():
    # 7 hexagons in a flower pattern
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # 6 surrounding
    sqrt3 = math.sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # 4 additional hexagons
    # Placed at distance 3, angles 30, 150, 210, 330
    # Oriented such that flat side faces radially out
    # Rotation angle = phi - 90
    add_angles = [30.0, 150.0, 210.0, 330.0]
    for phi in add_angles:
        rad = math.radians(phi)
        x = 3.0 * math.cos(rad)
        y = 3.0 * math.sin(rad)
        rot = phi - 90.0
        inner.append([x, y, rot])
        
    # Outer hexagon
    # Side length 4.0
    # Center (0,0)
    # Rotation 0.0
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
