# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

_DEADLINE = time.time() + 580.0


def _compute_min_side(center, angles):
    """Minimize side of a regular hexagon (fixed center, given angles) containing all 11 unit hexagons.

    Returns (side, feasible).
    """
    a = np.deg2rad(angles)
    normals = np.column_stack([np.cos(a), np.sin(a)])  # 6 x 2
    c = np.zeros(6)
    A_ub = np.zeros((66, 7))
    b_ub = np.zeros(66)
    for i in range(11):
        verts = center[2 * i: 2 * i + 2]
        for j in range(6):
            row = 6 * i + j
            A_ub[row, j] = normals[j, 0]
            A_ub[row, j + 6] = normals[j, 1]
            A_ub[row, 6] = 1.0
            b_ub[row] = -verts[0] * normals[j, 0] - verts[1] * normals[j, 1]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method="highs")
    if not res.success:
        return np.inf, False
    side = res.x[6]
    return side, True


def hexagon_packing_11():
    # 11 unit hexagons in a honeycomb cluster, scaled out by a small gap so no two touch.
    h = np.sqrt(3.0) / 2.0
    base_pos = [
        (0.0, 0.0),
        (1.5, h), (0.0, 2 * h), (-1.5, h),
        (-1.5, -h), (0.0, -2 * h), (1.5, -h),
        (3.0, 0.0), (0.0, 4 * h), (-3.0, 0.0), (0.0, -4 * h),
    ]
    # gap factor: scale positions so adjacent hexagons have a positive gap
    gap = 1.0 + 0.01  # 1% spacing increase -> clear separation
    center = np.array(base_pos, dtype=float) * gap

    base = np.zeros((11, 3))
    base[:, 0] = center[:, 0]
    base[:, 1] = center[:, 1]

    best_side = np.inf
    best_angle = 0.0

    for step in (2.0, 0.5, 0.1, 0.02, 0.005):
        if time.time() > _DEADLINE:
            break
        n_steps = int(60.0 / step)
        for k in range(n_steps):
            ang = k * step
            side, feas = _compute_min_side(base, ang * np.ones(6))
            if feas and side < best_side:
                best_side = side
                best_angle = ang

    inner_hex_data = base.copy()
    inner_hex_data[:, 2] = 0.0
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    return inner_hex_data, outer_hex_data, float(best_side)


# EVOLVE-BLOCK-END
