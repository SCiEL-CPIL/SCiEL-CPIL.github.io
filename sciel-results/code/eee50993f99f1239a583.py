import numpy as np

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Each inner hexagon may translate and rotate.
    """
    # Use a very small epsilon to ensure disjointness (separation between hexagons)
    # 1e-9 is safe for floating point precision while minimizing side length
    eps = 1e-9
    
    # Centers
    centers = []
    # Origin
    centers.append((0.0, 0.0))
    
    # 6 neighbors at dist sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
        
    # 4 more at dist 3 + 2*eps
    # Angles 30, 90, 150, 210
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    # Calculate required outer side length
    # Max X extent comes from ring 2 hexagons at 30 and 150 degrees
    # Center X = 3*cos(30) = 3*sqrt(3)/2
    # Max X offset = cos(30) = sqrt(3)/2
    # Total Max X = 3*sqrt(3)/2 + sqrt(3)/2 = 2*sqrt(3)
    # Max Y extent comes from ring 2 hexagon at 90 deg
    # Center Y = 3
    # Max Y offset = sin(90) = 1
    # Total Max Y = 4
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    
    # Outer hexagon rotated by 30 deg
    # Containment constraints:
    # s * sqrt(3)/2 >= max_x  => s >= 2 * max_x / sqrt(3)
    # s >= max_y
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    return inner, (0.0, 0.0), side, 30.0
