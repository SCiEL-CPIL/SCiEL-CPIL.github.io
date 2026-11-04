import numpy as np
import time

N = 11
DEADLINE = 140.0

def construct():
    """
    Optimal packing of 11 unit regular hexagons in a regular hexagon of side 4.
    
    Uses the "flower" packing: 1 center + 6 ring-1 + 4 ring-2 hexagons.
    All hexagons are flat-top (angle=30 degrees).
    """
    s3 = np.sqrt(3)
    eps = 1e-9
    angle_deg = 30.0
    
    centers = []
    
    # Ring 0: Center
    centers.append((0.0, 0.0))
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = s3 + eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    # Ring 2: 4 hexagons at distance 3 + 2*eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    # Compute required outer side length
    # Flat-top unit hexagon: max_x_offset = s3/2, max_y_offset = 1.0
    max_x = max(x + s3 / 2 for x, y in centers)
    max_y = max(y + 1.0 for x, y in centers)
    
    # Outer flat-top hexagon: width = s3*S, height = 2*S
    s_from_x = 2 * max_x / s3
    s_from_y = max_y
    side = max(s_from_x, s_from_y) + 1e-9
    
    inner = [[x, y, angle_deg] for x, y in centers]
    outer_center = (0.0, 0.0)
    outer_angle = angle_deg
    
    return inner, outer_center, side, outer_angle
