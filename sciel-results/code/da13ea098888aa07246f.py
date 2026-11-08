import math

N = 11

def construct():
    # 11 unit hexagons in a cross shape
    # Center at (0,0)
    # Axial coordinates:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (-2,0), (0,2), (0,-2)
    
    # Convert axial to cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    centers = []
    # Center
    centers.append((0.0, 0.0))
    
    # Ring 1
    ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q,r in ring1:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Ring 2 (tips)
    ring2 = [(2,0), (-2,0), (0,2), (0,-2)]
    for q,r in ring2:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Side length S = 2*sqrt(3) + 1
    S = 2 * math.sqrt(3) + 1
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, S, angle
