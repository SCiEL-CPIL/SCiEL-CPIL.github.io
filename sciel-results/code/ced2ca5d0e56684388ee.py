import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Container parameters
    L = 4.47
    center = (0.0, 0.0)
    container_angle = 0.0
    
    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. Ring 1: 6 hexagons
    d1 = math.sqrt(3)
    for i in range(6):
        angle = math.radians(i * 60)
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # 3. Ring 2: 4 hexagons
    d2 = 3.0
    for i in range(4):
        angle = math.radians(30 + i * 90)
        x = d2 * math.cos(angle)
        y = d2 * math.sin(angle)
        inner.append([x, y, 0.0])
        
    return inner, center, L, container_angle
# EVOLVE-BLOCK-END
