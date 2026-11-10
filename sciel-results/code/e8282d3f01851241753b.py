# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons inside a regular hexagon (same orientation),
    using a flower-of-7 plus 5 outer hexagons, then compute the minimal enclosing
    regular hexagon of that orientation via linear programming.

    Returns
    -------
    inner_hex_data : (12, 3) array of (x, y, angle_deg)
    outer_hex_data : (3,) array of (x, y, angle_deg)
    s : float, outer hexagon side length
    """
    # --- positions of the 12 unit hexagons (all axis-aligned, angle 0) ---
    # Flower of 7: center + 6 neighbors (honeycomb, hex-to-hex distance 3)
    positions = np.array([
        [0.0, 0.0],
        [3.0, 0.0],
        [-3.0, 0.0],
        [1.5, 2.598076211353316],
        [-1.5, 2.598076211353316],
        [1.5, -2.598076211353316],
        [-1.5, -2.598076211353316],
        # 5 more hexagons on the outside of three alternating vertices
        [6.0, 0.0],
        [4.5, 7.794228634059948],
        [-4.5, 7.794228634059948],
        [4.5, -7.794228634059948],
        [-4.5, -7.794228634059948],
    ])
    angles = np.zeros(12)

    # --- vertices of each unit hexagon (side 1, angle 0) ---
    k = np.arange(6)
    base_v = np.column_stack([np.cos(np.radians(60 * k)),
                              np.sin(np.radians(60 * k))])  # (6,2)
    all_vertices = positions[:, None, :] + base_v[None, :, :]  # (12,6,2)

    # --- outer hexagon orientation: same as inner (flat-top) ---
    # outward normals of the 6 outer edges, in vertex order (i=0..5)
    n_ang = np.radians(60 * k + 90)
    normals = np.column_stack([np.cos(n_ang), np.sin(n_ang)])  # (6,2)

    # --- LP: minimize s subject to normals @ v <= s + c for all vertices ---
    # variables: x = [s, cx, cy]
    A_ub = []
    b_ub = []
    for v in all_vertices.reshape(-1, 2):
        for n in normals:
            row = np.array([n[0], n[1], -1.0])
            A_ub.append(row)
            b_ub.append(v @ n)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    c_obj = np.array([1.0, 0.0, 0.0])
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

    if res.success:
        s, cx, cy = res.x
    else:
        # fallback
        s = 12.0
        cx, cy = 0.0, 0.0

    # Shift inner hexagons so that the outer hexagon is at the origin
    positions = positions - np.array([cx, cy])
    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    return inner_hex_data, outer_hex_data, float(s)


# EVOLVE-BLOCK-END
