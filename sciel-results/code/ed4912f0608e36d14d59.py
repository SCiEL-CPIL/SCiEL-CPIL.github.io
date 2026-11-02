import numpy as np

N = 12

def construct():
    # Define inner hexagons
    # 1 center
    centers = [(0.0, 0.0)]
    # 6 neighbors at distance sqrt(3)
    d1 = np.sqrt(3)
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    
    # 5 neighbors from 2nd layer, choosing those at distance 3 (angles 30, 90, 150, 210, 270)
    # Skip 330 to make it 5
    d2 = 3.0
    angles_2nd = [30, 90, 150, 210, 270]
    for deg in angles_2nd:
        angle = np.deg2rad(deg)
        centers.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        
    # Convert to list of [x, y, angle]
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    # Calculate minimal enclosing hexagon side length
    # Outer hexagon is flat-top, centered at (0,0)
    # Apothem directions: 30, 90, 150, 210, 270, 330 degrees
    apothem_angles = [np.deg2rad(30 + 60*i) for i in range(6)]
    
    max_apothem = 0.0
    s = 1.0
    # Vertices of a unit flat-top hexagon centered at (0,0)
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    for cx, cy, _ in inner:
        # Translate vertices to (cx, cy)
        verts = hex_verts + np.array([cx, cy])
        for v in verts:
            for a in apothem_angles:
                dist = v[0] * np.cos(a) + v[1] * np.sin(a)
                if dist > max_apothem:
                    max_apothem = dist
                    
    side = (2.0 / np.sqrt(3)) * max_apothem
    center = (0.0, 0.0)
    angle_outer = 0.0
    
    return inner, center, side, angle_outer
