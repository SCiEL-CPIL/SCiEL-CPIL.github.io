# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(np.arange(6) * 60)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_centers, outer_angle_deg):
    """
    Given inner hexagon centers (all unit hexagons, angle 0), find the minimum side length s
    of an outer regular hexagon with given angle that contains all inner hexagons.
    The outer hexagon is centered at origin.
    """
    all_vertices = []
    for cx, cy in inner_centers:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # shape (72, 2)

    apothem_factor = np.sqrt(3) / 2
    normal_angles = np.deg2rad(outer_angle_deg) + np.deg2rad(30) + np.deg2rad(np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    dots = all_vertices @ normals.T
    max_dot = np.max(dots)
    # Ensure s is positive
    s = max(max_dot / apothem_factor, 1e-9)
    return s


def _check_separation_simple(inner_centers, min_dist_sq):
    """Check if all pairs of centers are at least min_dist_sq apart."""
    n = len(inner_centers)
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner_centers[i][0] - inner_centers[j][0]
            dy = inner_centers[i][1] - inner_centers[j][1]
            if dx * dx + dy * dy < min_dist_sq:
                return False
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580

    sqrt3 = np.sqrt(3)
    # Scale factor to ensure strict separation (gap > 1e-6)
    # Min distance for touching is sqrt(3).
    # We need distance > sqrt(3) + 1e-6.
    # Scaling lattice by k makes distance k*sqrt(3).
    # k*sqrt(3) > sqrt(3) + 1e-6 => k > 1 + 1e-6/sqrt(3) ~ 1 + 5.77e-7.
    # Use 1.0001 to be safe.
    SCALE = 1.0001
    a1 = np.array([sqrt3 * SCALE, 0.0])
    a2 = np.array([sqrt3 / 2 * SCALE, 1.5 * SCALE])
    
    # Minimum squared distance for validity
    min_dist = sqrt3 + 1e-5
    min_dist_sq = min_dist * min_dist

    arrangements = []

    # Arrangement A: 4x3
    arrA = [(i, j) for j in range(3) for i in range(4)]
    arrangements.append(arrA)

    # Arrangement B: Hex cluster 1+6+5
    arrB = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2)]
    arrangements.append(arrB)

    # Arrangement C: 3-4-3-2
    arrC = []
    for j in range(4):
        if j == 0:
            for i in range(3): arrC.append((i, j))
        elif j == 1:
            for i in range(4): arrC.append((i, j))
        elif j == 2:
            for i in range(3): arrC.append((i, j))
        elif j == 3:
            for i in range(2): arrC.append((i, j))
    arrangements.append(arrC)

    # Arrangement D: 2-4-4-2
    arrD = []
    for j in range(4):
        if j == 0:
            for i in range(2): arrD.append((i, j))
        elif j in (1, 2):
            for i in range(4): arrD.append((i, j))
        elif j == 3:
            for i in range(2): arrD.append((i, j))
    arrangements.append(arrD)

    # Arrangement E: 3x4
    arrE = [(i, j) for j in range(4) for i in range(3)]
    arrangements.append(arrE)

    # Arrangement F: 4-3-3-2
    arrF = []
    for j in range(4):
        if j == 0:
            for i in range(4): arrF.append((i, j))
        elif j in (1, 2):
            for i in range(3): arrF.append((i, j))
        elif j == 3:
            for i in range(2): arrF.append((i, j))
    arrangements.append(arrF)

    # Arrangement G: 2-3-4-3
    arrG = []
    for j in range(4):
        if j == 0:
            for i in range(2): arrG.append((i, j))
        elif j == 1:
            for i in range(3): arrG.append((i, j))
        elif j == 2:
            for i in range(4): arrG.append((i, j))
        elif j == 3:
            for i in range(3): arrG.append((i, j))
    arrangements.append(arrG)
    
    # Arrangement H: 1+6+5 variant
    arrH = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    arrangements.append(arrH)

    best_s = float('inf')
    best_arrangement = None
    best_outer_angle = 0.0

    outer_angles_to_try = list(range(0, 60, 1))

    for arr in arrangements:
        centers = []
        for (i, j) in arr:
            x = i * a1[0] + j * a2[0]
            y = i * a1[1] + j * a2[1]
            centers.append([x, y])
        centers = np.array(centers)

        centroid = np.mean(centers, axis=0)
        centers = centers - centroid

        if not _check_separation_simple(centers, min_dist_sq):
            continue

        for oa in outer_angles_to_try:
            s = _min_containing_hex_side(centers, oa)
            if s < best_s:
                best_s = s
                best_arrangement = centers.copy()
                best_outer_angle = oa

        if time.time() > deadline:
            break

    if best_arrangement is None:
        # Fallback
        centers = []
        for (i, j) in arrangements[0]:
            x = i * a1[0] + j * a2[0]
            y = i * a1[1] + j * a2[1]
            centers.append([x, y])
        centers = np.array(centers)
        centroid = np.mean(centers, axis=0)
        centers = centers - centroid
        best_arrangement = centers
        best_s = _min_containing_hex_side(centers, 0.0)
        best_outer_angle = 0.0

    inner_hex_data = np.zeros((12, 3))
    for idx in range(12):
        inner_hex_data[idx, 0] = best_arrangement[idx, 0]
        inner_hex_data[idx, 1] = best_arrangement[idx, 1]
        inner_hex_data[idx, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, float(best_outer_angle)])

    return inner_hex_data, outer_hex_data, float(best_s)


# EVOLVE-BLOCK-END
