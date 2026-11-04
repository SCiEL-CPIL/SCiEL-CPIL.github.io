"""Improved hexagon packing: use hexagonal grid arrangement and optimize outer hexagon."""
import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon (flat-top by default at angle 0)."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180
    # Rotate
    cos_r = np.cos(angles + a)
    sin_r = np.sin(angles + a)
    xs = cx + side * cos_r
    ys = cy + side * sin_r
    return np.column_stack([xs, ys])

def _enclosing_hex_side(points, outer_angle_deg):
    """
    Given a set of points, find the minimum side length of a regular hexagon
    with rotation outer_angle_deg and center (0,0) that contains all points.
    
    A regular hexagon with side s, center (0,0), rotation theta has support
    in direction phi: s * (sqrt(3)/2) * |cos(phi - theta - k*pi/3)| ... 
    Actually, for a regular hexagon with flat-top orientation (angle 0), the
    apothem (distance from center to midpoint of each side) is s * sqrt(3)/2.
    The hexagon contains point (x,y) iff for all 6 side normals, the dot product
    of (x,y) with each normal is <= s * sqrt(3)/2.
    
    The 6 side normals for a hexagon with rotation theta are at angles
    theta + 30, theta + 90, theta + 150, theta + 210, theta + 270, theta + 330
    (for pointy-top) or theta + 0, 60, 120, 180, 240, 300 (for flat-top sides).
    
    Let me be careful: for a regular hexagon with vertices at angles 0,60,120,180,240,300
    (flat-top if we consider the top edge horizontal), the sides are perpendicular to
    directions 30, 90, 150, 210, 270, 330. The distance from center to each side is
    the apothem = s * sqrt(3)/2.
    
    So the hexagon contains all points iff max over all 6 normals of dot(point, normal) <= s*sqrt(3)/2.
    """
    theta = np.radians(outer_angle_deg)
    # Side normals at theta + 30, 90, 150, 210, 270, 330 degrees
    normal_angles = (theta + np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # For each point, compute dot product with each normal
    # We need the max over all points and all normals
    # points is (M, 2), normals is (6, 2)
    # Compute all dot products: (M, 6)
    dots = points @ normals.T
    max_dot = np.max(dots)
    # Also need to consider that points could be on either side, but since normals
    # come in opposite pairs, max over all 6 covers it.
    
    # s * sqrt(3)/2 >= max_dot
    # s >= 2 * max_dot / sqrt(3)
    side = 2.0 * max_dot / np.sqrt(3.0)
    return max(0.0, side)

def _compute_outer_side(inner_positions, inner_angles, outer_angle_deg):
    """Compute the enclosing hexagon side length for given inner hexagons."""
    all_points = []
    for (px, py), angle in zip(inner_positions, inner_angles):
        verts = _hex_vertices(px, py, 1.0, angle)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    return _enclosing_hex_side(all_points, outer_angle_deg)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Keep well under 150s
    
    # Strategy: Place 12 unit hexagons in a compact arrangement.
    # A hexagonal grid arrangement is natural. For 12 hexagons, we can use
    # a pattern like a larger hexagon of side 2 (which has 7 hexagons) plus 5 more,
    # or a linear/rectangular arrangement.
    
    # Let's try multiple candidate arrangements and optimize outer hexagon rotation.
    
    best_side = float('inf')
    best_config = None
    
    # Candidate 1: Linear arrangement along x-axis
    # Hexagons side by side: distance between centers = sqrt(3) for flat-top
    # Actually, for hexagons with side 1, the distance between adjacent centers
    # in a hex grid is 2 (for edge-sharing) or sqrt(3) (for vertex-adjacent in some configs).
    # For flat-top hexagons packed side by side, center-to-center distance is 2.
    
    # Let's use a honeycomb-like arrangement
    # Row 1: 6 hexagons, Row 2: 6 hexagons offset
    
    # Honeycomb spacing: for flat-top hexagons with side s=1:
    # Horizontal distance between centers in same row: 2*s = 2
    # Vertical distance between rows: sqrt(3)*s = sqrt(3)
    # Horizontal offset between rows: s = 1
    
    arrangements = []
    
    # Arrangement A: 2 rows of 6
    # Row 0: y = 0, x = 0, 2, 4, 6, 8, 10
    # Row 1: y = sqrt(3), x = 1, 3, 5, 7, 9, 11
    pos_a = []
    for i in range(6):
        pos_a.append((2.0 * i, 0.0))
    for i in range(6):
        pos_a.append((2.0 * i + 1.0, np.sqrt(3)))
    arrangements.append(('2x6', pos_a, [0.0] * 12))
    
    # Arrangement B: 3 rows of 4
    # Row 0: y = 0, x = 0, 2, 4, 6
    # Row 1: y = sqrt(3), x = 1, 3, 5, 7
    # Row 2: y = 2*sqrt(3), x = 0, 2, 4, 6
    pos_b = []
    for i in range(4):
        pos_b.append((2.0 * i, 0.0))
    for i in range(4):
        pos_b.append((2.0 * i + 1.0, np.sqrt(3)))
    for i in range(4):
        pos_b.append((2.0 * i, 2.0 * np.sqrt(3)))
    arrangements.append(('3x4', pos_b, [0.0] * 12))
    
    # Arrangement C: 4 rows of 3
    pos_c = []
    for i in range(3):
        pos_c.append((2.0 * i, 0.0))
    for i in range(3):
        pos_c.append((2.0 * i + 1.0, np.sqrt(3)))
    for i in range(3):
        pos_c.append((2.0 * i, 2.0 * np.sqrt(3)))
    for i in range(3):
        pos_c.append((2.0 * i + 1.0, 3.0 * np.sqrt(3)))
    arrangements.append(('4x3', pos_c, [0.0] * 12))
    
    # Arrangement D: Hexagonal cluster (center + 5 rings partially)
    # Center at (0,0), ring 1: 6 hexagons, ring 2: 6 more = 13, so take 12
    # Actually let's do: center + 5 from ring 1 + 1 from ring 2? Or just use the 3x4 which is more compact.
    
    # Let's also try rotating the inner hexagons
    for name, positions, angles in arrangements:
        # Center the arrangement at origin
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        centered_positions = [(p[0] - cx, p[1] - cy) for p in positions]
        
        # Try different outer hexagon rotations
        for outer_rot in range(0, 30, 5):  # 0, 5, 10, 15, 20, 25
            side = _compute_outer_side(centered_positions, angles, outer_rot)
            if side < best_side:
                best_side = side
                best_config = (centered_positions, angles, outer_rot)
        
        # Also try with inner hexagons rotated 30 degrees
        angles_rot = [30.0] * 12
        for outer_rot in range(0, 30, 5):
            side = _compute_outer_side(centered_positions, angles_rot, outer_rot)
            if side < best_side:
                best_side = side
                best_config = (centered_positions, angles_rot, outer_rot)
    
    # Check time
    if time.time() > deadline - 5.0:
        # Not enough time, use best so far
        pass
    else:
        # Try a few more arrangements with local optimization
        # For each arrangement, try small perturbations
        
        for name, positions, angles in arrangements:
            if time.time() > deadline - 5.0:
                break
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            base_positions = [(p[0] - cx, p[1] - cy) for p in positions]
            
            # Try small offsets
            for dx in [-0.2, 0.0, 0.2]:
                for dy in [-0.2, 0.0, 0.2]:
                    if time.time() > deadline - 5.0:
                        break
                    perturbed = [(p[0] + dx, p[1] + dy) for p in base_positions]
                    for outer_rot in range(0, 30, 10):
                        side = _compute_outer_side(perturbed, angles, outer_rot)
                        if side < best_side:
                            best_side = side
                            best_config = (perturbed, list(angles), outer_rot)
    
    if best_config is None:
        # Fallback
        positions = [(2.0 * i, 0.0) for i in range(12)]
        angles = [0.0] * 12
        outer_rot = 0.0
        best_side = _compute_outer_side(positions, angles, outer_rot)
        best_config = (positions, angles, outer_rot)
    
    positions, angles, outer_rot = best_config
    
    # The outer hexagon should be centered at the centroid of all inner hexagon vertices
    all_points = []
    for (px, py), angle in zip(positions, angles):
        verts = _hex_vertices(px, py, 1.0, angle)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    cx = np.mean(all_points[:, 0])
    cy = np.mean(all_points[:, 1])
    
    # Shift all inner hexagons so that the centroid is at origin
    # Then the outer hexagon is centered at (0,0)
    # But we need to report the outer center and inner positions consistently.
    # The task says: outer hexagon center (x, y), side length and rotation.
    # Inner rows are (x, y, angle_degrees).
    # So we can set outer center to (cx, cy) and keep inner positions as is.
    # Then compute side with points shifted by -(cx, cy).
    
    shifted_points = all_points - np.array([cx, cy])
    final_side = _enclosing_hex_side(shifted_points, outer_rot)
    
    inner = []
    for (px, py), angle in zip(positions, angles):
        inner.append([float(px), float(py), float(angle)])
    
    return inner, (float(cx), float(cy)), float(final_side), float(outer_rot)
