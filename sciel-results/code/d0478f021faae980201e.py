import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 7 core hexagons
    # Center at (0,0)
    # Neighbors at distance sqrt(3) at angles 30, 90, 150, 210, 270, 330
    sqrt3 = math.sqrt(3)
    
    inner = []
    
    # 1. Center
    inner.append([0.0, 0.0, 0.0])
    
    # 2-7. First shell (6 hexagons)
    for i in range(6):
        angle = math.radians(30 + i * 60)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 8-12. Second shell (5 hexagons)
    # We pick 5 from the 6 'good' positions (0, 60, 120, 180, 240, 300)
    # Centers at distance 3
    good_angles = [0, 60, 120, 180, 240]
    for deg in good_angles:
        angle = math.radians(deg)
        x = 3.0 * math.cos(angle)
        y = 3.0 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length 4.0 is sufficient for this configuration
    # Add small margin for safety
    side = 4.0 + 1e-6
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
