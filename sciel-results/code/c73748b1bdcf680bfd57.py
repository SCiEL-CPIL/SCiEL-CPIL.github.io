# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length (maximizing 1/s).
    
    Approach: Place 12 unit hexagons in a hexagonal tiling pattern (all same orientation),
    then compute the minimum enclosing regular hexagon (same orientation) by finding the
    maximum projection of all inner hexagon vertices onto the 6 edge-normal directions.
    """
    
    sqrt3 = np.sqrt(3)
    side = 1.0
    
    # Axial coordinates for a compact 12-hex cluster in a hex grid
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, -1), (-2, 1),
        (2, 0), (-2, 0),
        (1, 1), (-1, -1),
    ]
    
    # Convert axial to pixel coordinates for flat-top hexagons
    positions = []
    for q, r in axial_coords:
        x = 1.5 * side * q
        y = sqrt3 * side * (r + q / 2.0)
        positions.append((x, y))
    
    # All hexagons have angle 0
    inner_hex_data = np.array([(x, y, 0.0) for (x, y) in positions])
    
    # Compute all vertices of all inner hexagons
    all_vertices = []
    for (cx, cy, ang) in inner_hex_data:
        a_rad = np.radians(ang)
        for i in range(6):
            theta = a_rad + np.radians(60 * i)
            vx = cx + side * np.cos(theta)
            vy = cy + side * np.sin(theta)
            all_vertices.append((vx, vy))
    
    all_vertices = np.array(all_vertices)
    
    # The outer hexagon has angle 0, centered at origin
    # Edge normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = [np.radians(30 + 60 * i) for i in range(6)]
    
    # Find maximum projection in each normal direction
    max_projections = []
    for na in normal_angles:
        nx, ny = np.cos(na), np.sin(na)
        projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
        max_projections.append(np.max(projections))
    
    apothem = max(max_projections)
    outer_hex_side_length = 2.0 * apothem / sqrt3
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
