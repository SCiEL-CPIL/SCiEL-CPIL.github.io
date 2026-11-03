import numpy as np

def construct():
    # Use a very small epsilon to ensure disjointness while minimizing side length
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
    # Max X extent comes from ring 2 hexagons at 30 and 150 degrees?
    # Ring 2 at 30 deg: Center X = (3+2eps)*cos(30) = (3+2eps)*sqrt(3)/2.
    # Rightmost point of that hex: Center X + sqrt(3)/2.
    # Total Max X = (3+2eps)*sqrt(3)/2 + sqrt(3)/2 = (4+2eps)*sqrt(3)/2 = (2+eps)*sqrt(3).
    
    # Max Y extent comes from ring 2 hexagon at 90 deg.
    # Center Y = 3+2eps.
    # Topmost point: Vertex at 90 deg?
    # v1: (cos90, sin90) = (0, 1).
    # So max y offset is 1.
    # Total Max Y = 3+2eps + 1 = 4+2eps.
    
    # Outer hexagon orientation: 30 degrees (flat top/bottom, vertical sides).
    # Width = sqrt(3) * S.
    # Height = 2 * S.
    
    # Constraints:
    # sqrt(3) * S >= 2 * Max X
    # 2 * S >= 2 * Max Y  => S >= Max Y
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    
    side = max(s_x, s_y)
    
    # Add a tiny margin to side to ensure containment
    side += 1e-9
    
    return inner, (0.0, 0.0), side, 30.0
