# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3.0)
    
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, sqrt3, 0.0],
        [-1.0, sqrt3, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, -sqrt3, 0.0],
        [1.0, -sqrt3, 0.0],
        [3.0, sqrt3, 0.0],
        [-3.0, sqrt3, 0.0],
        [-3.0, -sqrt3, 0.0],
        [3.0, -sqrt3, 0.0],
    ])
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    all_verts = []
    for (cx, cy, ang) in inner_hex_data:
        a_rad = np.radians(ang)
        for i in range(6):
            theta = a_rad + np.radians(60 * i)
            vx = cx + np.cos(theta)
            vy = cy + np.sin(theta)
            all_verts.append((vx, vy))
    all_verts = np.array(all_verts)
    
    max_s = 0.0
    for (vx, vy) in all_verts:
        vals = [
            np.sqrt(3)/2 * vx + 0.5 * vy,
            -0.5 * vx + np.sqrt(3)/2 * vy,
            -np.sqrt(3)/2 * vx + 0.5 * vy,
            -np.sqrt(3)/2 * vx - 0.5 * vy,
            0.5 * vx - np.sqrt(3)/2 * vy,
            np.sqrt(3)/2 * vx - 0.5 * vy,
        ]
        max_s = max(max_s, max(vals))
    
    outer_hex_side_length = max_s
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
