# EVOLVE-BLOCK-START
import numpy as np


def _hexagon_vertices(cx, cy, side, angle_deg):
    """Compute the 6 vertices of a regular hexagon."""
    angle = np.deg2rad(angle_deg)
    verts = np.zeros((6, 2))
    for i in range(6):
        theta = angle + np.deg2rad(60 * i)
        verts[i] = [cx + side * np.cos(theta), cy + side * np.sin(theta)]
    return verts


def _point_in_hexagon(px, py, center, side, angle_deg):
    """Check if point (px, py) is inside hexagon with given center, side, angle."""
    verts = _hexagon_vertices(center[0], center[1], side, angle_deg)
    for j in range(6):
        p1 = verts[j]
        p2 = verts[(j + 1) % 6]
        edge = p2 - p1
        to_pt = np.array([px, py]) - p1
        cross = edge[0] * to_pt[1] - edge[1] * to_pt[0]
        if cross < -1e-6:
            return False
    return True


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, maximizing 1/s.
    Uses a compact 3x4 honeycomb arrangement.
    Returns (inner, outer, s).
    """
    # Honeycomb grid: 3 rows of 4 hexes each
    # Horizontal spacing between adjacent centers in same row: sqrt(3) * side = sqrt(3)
    # Vertical spacing between rows: 1.5 * side = 1.5 (for side=1)
    dx = np.sqrt(3)
    dy = 1.5

    positions = []
    # Row 0: 4 hexes at y=0
    for i in range(4):
        positions.append([i * dx, 0.0, 0.0])
    # Row 1: 4 hexes at y=dy, offset by dx/2
    for i in range(4):
        positions.append([i * dx + dx / 2, dy, 0.0])
    # Row 2: 4 hexes at y=2*dy
    for i in range(4):
        positions.append([i * dx, 2 * dy, 0.0])

    pos_arr = np.array(positions, dtype=float)
    # Center the arrangement at origin
    centroid = pos_arr[:, :2].mean(axis=0)
    pos_arr[:, :2] -= centroid

    inner_hex_data = pos_arr

    # Find minimal outer hexagon side length s that contains all inner hexagons
    best_s = 10.0
    found = False
    for s in np.linspace(3.0, 8.0, 501):
        valid = True
        for row in inner_hex_data:
            cx, cy, ang = row[0], row[1], row[2]
            inner_verts = _hexagon_vertices(cx, cy, 1.0, ang)
            for vx, vy in inner_verts:
                if not _point_in_hexagon(vx, vy, [0.0, 0.0], s, 0.0):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            best_s = s
            found = True
            break

    if not found:
        best_s = 8.0

    # Add small margin for numerical safety
    best_s += 0.01

    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END
