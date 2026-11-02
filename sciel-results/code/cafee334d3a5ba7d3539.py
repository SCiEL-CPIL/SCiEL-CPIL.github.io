import numpy as np

def construct():
    # Define 11 unit hexagons in a compact cluster
    # Center + 6 neighbors + 4 from next ring
    sqrt3 = np.sqrt(3)
    
    # Positions (x, y, angle_degrees)
    inner = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, sqrt3, 0.0],
        [-1.0, sqrt3, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -sqrt3, 0.0],
        [1.0, -sqrt3, 0.0],
        [4.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0],
        [2.0, 2*sqrt3, 0.0],
        [-2.0, 2*sqrt3, 0.0]
    ]
    
    # Calculate minimal side length S for outer hexagon centered at (0,0) with angle 0
    # Outer hexagon vertices at angles 0, 60, ...
    # Sides are perpendicular to normals at 30, 90, ...
    
    max_proj = 0.0
    # Normals for outer hexagon (angle 0)
    normal_angles = [np.deg2rad(30 + i * 60) for i in range(6)]
    normals = [(np.cos(a), np.sin(a)) for a in normal_angles]
    
    for x, y, ang in inner:
        # Get vertices of this inner hexagon
        rad = np.deg2rad(ang)
        hex_angles = np.arange(6) * np.pi / 3 + rad
        vx = x + np.cos(hex_angles)
        vy = y + np.sin(hex_angles)
        
        for i in range(6):
            px, py = vx[i], vy[i]
            for nx, ny in normals:
                proj = px * nx + py * ny
                if proj > max_proj:
                    max_proj = proj
    
    # S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    side = (2.0 * max_proj) / sqrt3
    
    # Add a tiny epsilon to ensure strict containment if needed, though usually >= is fine
    side += 1e-9
    
    return inner, (0.0, 0.0), side, 0.0
