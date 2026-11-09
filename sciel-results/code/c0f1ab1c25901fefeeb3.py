import numpy as np
import time
from itertools import combinations, product

N = 11
SQRT3 = np.sqrt(3)

def hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = np.array([
        [cx + s * np.cos(a + i * np.pi / 3),
         cy + s * np.sin(a + i * np.pi / 3)]
        for i in range(6)
    ])
    return verts

def point_in_hex(px, py, cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    for i in range(6):
        theta = i * np.pi / 3
        if lx * np.cos(theta) + ly * np.sin(theta) > s * SQRT3 / 2 + 1e-12:
            return False
    return True

def _point_in_poly(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _seg_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def hexagons_disjoint(v1, v2, s, eps=1e-9):
    for v in v1:
        if _point_in_poly(v[0], v[1], v2):
            return False
    for v in v2:
        if _point_in_poly(v[0], v[1], v1):
            return False
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            if _seg_intersect(p1, p2, p3, p4):
                return False
    return True

def _outer_side_for_config(centers, outer_angle, center_xy):
    """Compute the minimum outer side length for given inner centers and outer hex params."""
    all_verts = np.vstack([hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers])
    a = np.radians(outer_angle)
    normals = np.array([[np.cos(a + np.pi / 6 + i * np.pi / 3),
                         np.sin(a + np.pi / 6 + i * np.pi / 3)] for i in range(6)])
    dx = all_verts[:, 0] - center_xy[0]
    dy = all_verts[:, 1] - center_xy[1]
    proj_matrix = np.column_stack((dx, dy)) @ normals.T
    half_widths = np.maximum(proj_matrix.max(axis=0), -proj_matrix.min(axis=0))
    side = half_widths.max() / (SQRT3 / 2)
    return side

def _find_best_outer(centers):
    """Find the best outer hexagon parameters for a given set of inner centers."""
    all_verts = np.vstack([hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers])
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()

    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0

    # Coarse search
    for outer_angle in np.linspace(0, 60, 121):
        side = _outer_side_for_config(centers, outer_angle, (cx_mean, cy_mean))
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle

    # Fine search
    if best_side < float('inf'):
        for outer_angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 201):
            side = _outer_side_for_config(centers, outer_angle, (cx_mean, cy_mean))
            if side < best_side:
                best_side = side
                best_center = (cx_mean, cy_mean)
                best_angle = outer_angle

    # Try a few candidate centers
    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + np.pi / 6 + i * np.pi / 3),
                         np.sin(a + np.pi / 6 + i * np.pi / 3)] for i in range(6)])

    candidates = [
        (cx_mean, cy_mean),
        ((all_verts[:, 0].min() + all_verts[:, 0].max()) / 2,
         (all_verts[:, 1].min() + all_verts[:, 1].max()) / 2)
    ]

    for trial_cx, trial_cy in candidates:
        side = _outer_side_for_config(centers, best_angle, (trial_cx, trial_cy))
        if side < best_side:
            best_side = side
            best_center = (trial_cx, trial_cy)

    return best_side, best_center, best_angle

def _axial_to_pixel(q, r):
    x = 1.5 * q
    y = SQRT3 * (r + 0.5 * q)
    return (x, y)

def _check_validity(centers):
    """Check if inner hexagons are all disjoint."""
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in centers]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if not hexagons_disjoint(inner_verts[i], inner_verts[j], 1.0):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    # Base 7 hexagons: center + 6 neighbors
    base_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    base_centers = [_axial_to_pixel(q, r) for q, r in base_axial]

    # Layer 2 positions (12 positions at distance 2)
    layer2_axial = [
        (2, 0), (2, -1), (2, -2),
        (1, -2), (0, -2), (-1, -1),
        (-2, 0), (-2, 1), (-2, 2),
        (-1, 2), (0, 2), (1, 1)
    ]

    # We need to pick 4 from layer 2. But not all combinations are valid
    # (some layer-2 hexagons overlap with each other or with base hexagons)
    # Let's pre-filter: for each layer-2 position, check if it's disjoint from all base hexagons
    # and from each other

    # First, let's just try a few promising configurations
    # The most compact 11-hexagon configurations are known. Let's try symmetric ones.

    # Configuration 1: The one in the current program
    config1_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (1, -2), (0, -2)]
    config1_centers = [_axial_to_pixel(q, r) for q, r in config1_axial]

    # Configuration 2: Different selection of 4 from layer 2
    config2_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (1, -2), (-1, 2)]
    config2_centers = [_axial_to_pixel(q, r) for q, r in config2_axial]

    # Configuration 3: Another selection
    config3_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (1, -2), (0, 2)]
    config3_centers = [_axial_to_pixel(q, r) for q, r in config3_axial]

    # Configuration 4: Try a more symmetric arrangement
    config4_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (1, -2), (0, -2), (-1, -1)]
    config4_centers = [_axial_to_pixel(q, r) for q, r in config4_axial]

    # Configuration 5: Yet another
    config5_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (1, -2), (-2, 1)]
    config5_centers = [_axial_to_pixel(q, r) for q, r in config5_axial]

    # Configuration 6: Try the "flower" with 4 opposite pairs
    config6_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (-2, 0), (-2, 1)]
    config6_centers = [_axial_to_pixel(q, r) for q, r in config6_axial]

    # Configuration 7: Another symmetric one
    config7_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (0, -2), (-2, 1)]
    config7_centers = [_axial_to_pixel(q, r) for q, r in config7_axial]

    # Configuration 8: Try 4 in a line
    config8_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                     (2, 0), (2, -1), (2, -2), (1, -2)]
    config8_centers = [_axial_to_pixel(q, r) for q, r in config8_axial]

    configs = [
        config1_centers,
        config2_centers,
        config3_centers,
        config4_centers,
        config5_centers,
        config6_centers,
        config7_centers,
        config8_centers,
    ]

    # Also try some non-lattice arrangements: slightly perturb positions
    # and see if we can get a better fit

    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_centers = config1_centers

    for centers in configs:
        if time.time() > deadline:
            break

        # Check validity
        if not _check_validity(centers):
            continue

        # Find best outer hexagon
        side, center, angle = _find_best_outer(centers)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = angle
            best_centers = centers

    # If we have time, try to optimize by slightly moving the layer-2 hexagons
    if time.time() < deadline - 30:
        # Try small perturbations of the best configuration
        base = best_centers
        perturbation_steps = [0.0, 0.01, 0.02, 0.05, 0.1]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7),
                      (0.7, -0.7), (-0.7, -0.7)]

        for step in perturbation_steps:
            if time.time() > deadline:
                break
            for dx, dy in directions:
                if time.time() > deadline:
                    break
                # Perturb only the layer-2 hexagons (indices 7-10)
                new_centers = list(base)
                for i in range(7, 11):
                    cx, cy = new_centers[i]
                    new_centers[i] = (cx + step * dx, cy + step * dy)

                if not _check_validity(new_centers):
                    continue

                side, center, angle = _find_best_outer(new_centers)
                if side < best_side:
                    best_side = side
                    best_center = center
                    best_angle = angle
                    best_centers = new_centers

    # Try different inner hexagon orientations
    if time.time() < deadline - 20:
        # Try rotating all inner hexagons by 30 degrees
        for rot in [0.0, 30.0, 60.0]:
            if time.time() > deadline:
                break
            # For now, just try the same positions but with rotated hexagons
            # This requires recomputing vertices, so we skip for now
            pass

    # Build inner list
    inner = [[float(cx), float(cy), float(0.0)] for cx, cy in best_centers]

    # Final verification and safety margin
    inner_verts = [hex_vertices(cx, cy, 1.0, 0.0) for cx, cy in best_centers]
    all_check_verts = np.vstack(inner_verts)

    a = np.radians(best_angle)
    normals = np.array([[np.cos(a + np.pi / 6 + i * np.pi / 3),
                         np.sin(a + np.pi / 6 + i * np.pi / 3)] for i in range(6)])

    dx = all_check_verts[:, 0] - best_center[0]
    dy = all_check_verts[:, 1] - best_center[1]

    max_violation = 0
    for i in range(6):
        proj = dx * normals[i, 0] + dy * normals[i, 1]
        max_violation = max(max_violation, proj.max() - best_side * SQRT3 / 2)
        max_violation = max(max_violation, -proj.min() - best_side * SQRT3 / 2)

    if max_violation > 1e-9:
        best_side += max_violation / (SQRT3 / 2) + 1e-8

    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
