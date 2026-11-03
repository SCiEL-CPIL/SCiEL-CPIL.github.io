import math

N = 11

def construct():
    # Axial coordinates (q, r)
    coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    inner = []
    for q, r in coords:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    # Side length calculation
    # Max projection was 3.
    # L = (2/sqrt(3)) * 3 + 1
    side = 2 * math.sqrt(3) + 1.0
    angle = 0.0
    
    return inner, center, side, angle
