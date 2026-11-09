import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Center hexagon
    inner = [[0.0, 0.0, 0.0]]
    
    # 2. 6 surrounding hexagons
    dist1 = math.sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = dist1 * math.cos(rad)
        y = dist1 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # 3. 4 additional hexagons at "diagonal" positions (distance 3)
    # The 6 diagonal positions are at angles 30, 90, 150, 210, 270, 330
    # We pick 4 of them to maintain symmetry and minimize bounding box
    # Picking 30, 150, 210, 330 gives a 180-degree symmetric shape
    dist2 = 3.0
    angles_to_use = [30, 150, 210, 330]
    for angle in angles_to_use:
        rad = math.radians(angle)
        x = dist2 * math.cos(rad)
        y = dist2 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # The max distance from center to any vertex of the inner hexagons is 4.0
    # (Distance 3.0 to center of outer hex + 1.0 to vertex)
    # A regular hexagon with side length S has radius S.
    # So S = 4.0
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
