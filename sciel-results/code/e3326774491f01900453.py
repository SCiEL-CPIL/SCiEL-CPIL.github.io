import math

N = 11

def construct():
    # Centers of 11 unit hexagons in a hexagonal grid
    # Center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors at distance 1 (axial)
    # Axial to Cartesian: x = 1.5 * q, y = sqrt(3) * (r + 0.5 * q)
    neighbors = [
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    ]
    for q, r in neighbors:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    # 4 from distance 2 (axial)
    # Selected for symmetry: (2,0), (0,2), (-2,0), (0,-2)
    dist2 = [
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    for q, r in dist2:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        centers.append((x, y))
        
    inner = []
    max_radius = 0.0
    
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        # Calculate farthest distance from origin for this hexagon
        # The farthest point is at distance 1 from the center, in the radial direction
        # (since the hexagon is convex and contains the origin in its "direction")
        # Actually, for a hexagon at (cx, cy), the max distance from origin is sqrt(cx^2+cy^2) + 1
        # This is an upper bound. The actual max distance depends on orientation.
        # For flat-top hexagons, the vertices are at angles 0, 60, ...
        # The radial direction might not align with a vertex.
        # However, the max distance from origin to any point in the hexagon is:
        # max over vertices v of |v|
        # Let's just compute the max distance to the 6 vertices.
        for i in range(6):
            angle = math.radians(60 * i)
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            dist = math.sqrt(vx*vx + vy*vy)
            if dist > max_radius:
                max_radius = dist
                
    # The outer hexagon must have side length at least max_radius
    # (Circumradius of regular hexagon = side length)
    side = max_radius
    
    return inner, (0.0, 0.0), side, 0.0
