"""Unit regular hexagons inside a regular hexagon — improved packing via hexagonal cluster."""
import math
import numpy as np

N = 11

def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon (flat-top by default, rot_deg rotates CCW)."""
    angles = np.deg2rad([0, 60, 120, 180, 240, 300] + rot_deg)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return list(zip(xs, ys))

def _min_enclosing_hexagon_side(centers, rotations_deg, unit_side=1.0):
    """
    Find the minimal side length S of a regular hexagon (centered at origin, flat-top, rot 0)
    that contains all the given unit hexagons (each rotated by rotations_deg).
    
    For a flat-top hexagon centered at origin with side S, the boundary is:
      x <= S * (2*sqrt(3)/3)  ->  x <= S * 2/sqrt(3)  (right side, normal at 0 deg)
      x * cos(60) + y * sin(60) <= S * 2/sqrt(3)  (top-right side, normal at 60 deg)
      ... etc for 6 directions at 0, 60, 120, 180, 240, 300 degrees.
    
    Actually, the apothem of the outer hexagon is a = S * sqrt(3)/2.
    The constraint for each inner vertex (vx, vy) in direction theta:
      vx * cos(theta) + vy * sin(theta) <= a  for all 6 theta in {0, 60, ..., 300}
    
    So a_min = max over all inner vertices and all 6 directions of (vx*cos(theta) + vy*sin(theta))
    Then S = 2 * a_min / sqrt(3).
    """
    thetas = np.deg2rad([0, 60, 120, 180, 240, 300])
    max_proj = 0.0
    for (cx, cy), rot in zip(centers, rotations_deg):
        verts = _hex_vertices(cx, cy, unit_side, rot)
        for (vx, vy) in verts:
            for theta in thetas:
                proj = vx * math.cos(theta) + vy * math.sin(theta)
                if proj > max_proj:
                    max_proj = proj
    a_min = max_proj
    S = 2.0 * a_min / math.sqrt(3)
    return S

def construct():
    # Hexagonal grid coordinates for unit hexagons (flat-top orientation)
    # In a hexagonal grid with "pointy" spacing:
    #   dx = sqrt(3) * side, dy = 1.5 * side for adjacent rows
    # For side=1, flat-top hexagons:
    #   Horizontal spacing: sqrt(3)
    #   Vertical spacing: 1.5
    #   Offset for odd rows: sqrt(3)/2
    
    sqrt3 = math.sqrt(3)
    
    # Build a compact cluster of 11 hexagons:
    # Core: 7 hexagons (1 center + 6 neighbors)
    # Then 4 more hexagons added to extend the cluster in a compact shape.
    
    # Hex grid axial coordinates (q, r) to cartesian (x, y) for flat-top:
    #   x = sqrt(3) * (q + r/2)
    #   y = 3/2 * r
    # where r is the row, q is the column offset.
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    # 7-hexagon cluster: center (0,0) + 6 neighbors
    core_axial = [(0, 0),
                  (1, 0), (1, -1), (0, -1),
                  (-1, 0), (-1, 1), (0, 1)]
    
    # Add 4 more hexagons to make 11.
    # To minimize the diameter, extend in a way that keeps the shape compact.
    # Option: add hexagons at positions that fill in a "bump" pattern.
    # Let's try adding at: (2, 0), (2, -1), (-2, 0), (-2, 1) — this makes a wider shape.
    # Or: (1, -2), (2, -1), (-1, 2), (-2, 1) — diagonal extensions.
    # Let's try a more compact arrangement:
    # Add: (2, 0), (2, -1), (-2, 0), (-2, 1)
    extra_axial = [(2, 0), (2, -1), (-2, 0), (-2, 1)]
    
    all_axial = core_axial + extra_axial
    centers = [axial_to_cartesian(q, r) for (q, r) in all_axial]
    rotations = [0.0] * N
    
    # Compute minimal enclosing hexagon side length
    S = _min_enclosing_hexagon_side(centers, rotations)
    
    # Return format: inner = [[x, y, angle_degrees], ...], center, side, angle
    inner = [[cx, cy, rot] for (cx, cy), rot in zip(centers, rotations)]
    return inner, (0.0, 0.0), S, 0.0
