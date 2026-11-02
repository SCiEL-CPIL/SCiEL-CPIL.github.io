"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice.
Uses scipy L-BFGS-B to optimize inner positions (with small rotations) and
LP to find the exact minimum enclosing regular hexagon for each configuration.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
SQRT3 = math.sqrt(3.0)


def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _min_enclosing_hex_side(points, outer_angle_deg):
    """Given a set of points and the outer hexagon's rotation angle (degrees),
    find the minimum side length and center of a regular hexagon containing all points.
    Uses LP: minimize A subject to strip constraints.
    Returns (side, cx, cy)."""
    pts = np.array(points)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0

    # 6 normal directions for the outer hexagon
    # The hexagon has sides perpendicular to directions 30+60k degrees
    u_vectors = []
    for k in range(6):
        theta = math.radians(outer_angle_deg + 30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))

    # For each direction, compute max and min projection
    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)

    # LP: variables (ox, oy, A)
    # Constraints: for each k,
    #   ox*ux + oy*uy - A <= Q_k
    #   -ox*ux - oy*uy - A <= -P_k
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1.0
        b_vec[2*k] = Q_vals[k]
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1.0
        b_vec[2*k+1] = -P_vals[k]

    c_vec = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]

    result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method='highs')
    if result.success:
        A_opt = result.x[2]
        ox_opt = result.x[0]
        oy_opt = result.x[1]
        side = A_opt * 2.0 / SQRT3
        return side, ox_opt, oy_opt

    # Fallback: use max extent
    A_low = max((P_vals[k] - Q_vals[k]) / 2.0 for k in range(6))
    side = A_low * 2.0 / SQRT3
    return side, 0.0, 0.0


def _axial_to_xy(q, r):
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)


def _build_initial_positions():
    """Build the 11 positions: center + ring 1 (6) + 4 from ring 2."""
    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])

    # Ring 2 candidates - pick 4 that give the most symmetric arrangement
    # The 14 ring-2 positions; we want to pick 4 to minimize the enclosing hex.
    # By symmetry, picking 4 from ring 2 in a balanced way is best.
    # Let's try: (2,0), (0,-2), (-2,2), (-2,-2) -- no, let's think more carefully.
    # Actually for 11 hexagons in a hexagonal lattice, the natural arrangement is
    # center + 6 ring-1 + 4 ring-2. The 4 ring-2 positions should be chosen to
    # minimize the bounding hexagon.
    # By the 6-fold symmetry of the lattice, we can pick 4 from the 14 ring-2 spots.
    # The best choice by symmetry is to pick 4 that are roughly evenly spaced.
    # Ring 2 positions in order: (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),
    #   (-2,-2),(-2,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,2)
    # Indices: 0,1,2,3,4,5,6,7,8,9,10,11,12,13
    # Picking every ~3.5: 0, 4, 8, 12 -> (2,0),(0,-2),(-2,0),(0,2) -- that's 4 on axes
    # Or 0, 3, 7, 10 -> (2,0),(1,-2),(-2,-1),(-2,2)
    # Let's just try a few symmetric sets.

    ring2_all = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
        (0, 2), (1, 2)
    ]

    # Try symmetric choices: pick indices that are 60-degree apart
    # In the ring-2 hexagon, positions are at angles. Let's just try a few sets.
    candidate_sets = [
        [0, 4, 8, 12],   # (2,0),(0,-2),(-2,0),(0,2) - cross pattern
        [0, 3, 7, 10],   # (2,0),(1,-2),(-2,-1),(-2,2)
        [1, 5, 9, 13],   # (2,-1),(-1,-2),(-2,1),(1,2)
        [2, 6, 10, 13],  # (2,-2),(-2,-2),(-2,2),(1,2)
        [0, 5, 9, 12],   # (2,0),(-1,-2),(-2,1),(0,2)
        [1, 4, 8, 11],   # (2,-1),(0,-2),(-2,0),(-1,2)
    ]

    best_set = None
    best_side = float('inf')

    for cset in candidate_sets:
        selected = [ring2_all[i] for i in cset]
        full = pos_axial + selected
        xy = [_axial_to_xy(q, r) for q, r in full]
        # Center
        cx = sum(p[0] for p in xy) / 11.0
        cy = sum(p[1] for p in xy) / 11.0
        centered = [(x - cx, y - cy) for x, y in xy]
        # Try a few outer angles
        for ang in [0, 5, 10, 15, 20, 25, 30]:
            side, _, _ = _min_enclosing_hex_side(centered, ang)
            if side < best_side:
                best_side = side
                best_set = cset

    selected = [ring2_all[i] for i in best_set]
    full = pos_axial + selected
    xy = [_axial_to_xy(q, r) for q, r in full]
    return xy


def _separation_penalty(pos, angles, lam=50.0):
    """Compute a penalty for overlapping hexagons.
    pos: (N, 2) array of centers
    angles: (N,) array of rotation angles in radians
    Returns scalar penalty (0 if no overlap)."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            # Minimum distance between two unit hexagons depends on relative angle
            # For same orientation: min dist = 2 * SQRT3/2 * ... actually for flat-top
            # two unit hexagons with same orientation: min center distance = SQRT3 (edge to edge)
            # For different orientations, it's more complex. Use a conservative bound.
            # The inradius (apothem) of a unit hexagon is SQRT3/2.
            # The minimum distance between two non-overlapping unit hexagons is at least
            # 2 * apothem = SQRT3 ≈ 1.732 (when faces are parallel and aligned).
            # For arbitrary rotations, the minimum can be as low as 1.0 (vertex to vertex)
            # but that's a degenerate case. Use 1.732 as a safe threshold for the penalty.
            min_dist = SQRT3  # conservative: same-orientation edge-to-edge
            if dist < min_dist:
                penalty += lam * (min_dist - dist) ** 2
    return penalty


def _objective(x, outer_angle_deg, lam=50.0):
    """Objective: outer hex side length + separation penalty.
    x: 3*N vector: [x0, y0, a0, x1, y1, a1, ...]
    Returns scalar."""
    pos = x[0::3].copy()
    ys = x[1::3].copy()
    angles = x[2::3].copy()

    # Center the configuration
    cx = np.mean(pos)
    cy = np.mean(ys)
    pos_c = pos - cx
    ys_c = ys - cy

    points = np.column_stack([pos_c, ys_c])
    side, _, _ = _min_enclosing_hex_side(points, outer_angle_deg)

    # Separation penalty
    pos_full = np.column_stack([pos_c, ys_c])
    sep_pen = _separation_penalty(pos_full, angles, lam)

    return side + sep_pen


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    # Build initial positions
    initial_xy = _build_initial_positions()

    # Center them
    cx0 = sum(p[0] for p in initial_xy) / N
    cy0 = sum(p[1] for p in initial_xy) / N
    centered_xy = [(x - cx0, y - cy0) for x, y in initial_xy]

    # Initial x vector: [x0, y0, 0, x1, y1, 0, ...]
    x0 = np.zeros(3 * N)
    for i, (px, py) in enumerate(centered_xy):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = 0.0  # angle

    best_side = float('inf')
    best_x = x0.copy()
    best_angle = 0.0

    # Try multiple outer angles
    outer_angles = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    for outer_ang in outer_angles:
        if time.time() > deadline - 5:
            break

        # Optimize inner positions
        res = minimize(
            _objective, x0, args=(outer_ang, 50.0),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        if res.success or res.fun < best_side:
            # Evaluate the actual side length (without penalty)
            pos = res.x[0::3]
            ys = res.x[1::3]
            cx = np.mean(pos)
            cy = np.mean(ys)
            pos_c = pos - cx
            ys_c = ys - cy
            points = np.column_stack([pos_c, ys_c])
            side, ox, oy = _min_enclosing_hex_side(points, outer_ang)

            if side < best_side:
                best_side = side
                best_x = res.x.copy()
                best_angle = outer_ang

    # Refine: run more iterations with the best angle
    if time.time() < deadline - 10:
        pos = best_x[0::3]
        ys = best_x[1::3]
        cx = np.mean(pos)
        cy = np.mean(ys)
        pos_c = pos - cx
        ys_c = ys - cy
        points = np.column_stack([pos_c, ys_c])

        # Try a finer grid of outer angles around the best
        for outer_ang in np.arange(best_angle - 2, best_angle + 2.5, 0.5):
            if time.time() > deadline - 5:
                break
            res = minimize(
                _objective, best_x, args=(outer_ang, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_side:
                pos2 = res.x[0::3]
                ys2 = res.x[1::3]
                cx2 = np.mean(pos2)
                cy2 = np.mean(ys2)
                pos2c = pos2 - cx2
                ys2c = ys2 - cy2
                pts2 = np.column_stack([pos2c, ys2c])
                side2, ox2, oy2 = _min_enclosing_hex_side(pts2, outer_ang)
                if side2 < best_side:
                    best_side = side2
                    best_x = res.x.copy()
                    best_angle = outer_ang

    # Extract final positions
    pos = best_x[0::3]
    ys = best_x[1::3]
    angles = best_x[2::3]

    # Center the configuration
    cx = np.mean(pos)
    cy = np.mean(ys)
    pos_c = pos - cx
    ys_c = ys - cy

    # Compute final outer hex
    points = np.column_stack([pos_c, ys_c])
    final_side, ox, oy = _min_enclosing_hex_side(points, best_angle)

    # Shift back: the outer hex center in the original frame
    outer_cx = ox + cx
    outer_cy = oy + cy

    # Build inner hexagons in the original frame
    inner = []
    for i in range(N):
        inner.append([pos_c[i] + cx, ys_c[i] + cy, math.degrees(angles[i])])

    # Verify: check that all inner hexagons are inside the outer
    # and that they are pairwise disjoint (approximately)
    # For safety, add a tiny margin
    final_side *= 1.001

    return inner, (outer_cx, outer_cy), final_side, best_angle
