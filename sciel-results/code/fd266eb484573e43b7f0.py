# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, a_deg):
    """Vertices of a regular hexagon, center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(th), cy + d * np.sin(th)])


def _edge_normals(d, a_deg):
    """6 outward unit edge normals for a regular hexagon (side d, angle a_deg)."""
    verts = _hex_vertices(0.0, 0.0, d, a_deg)
    normals = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        n = n / (np.linalg.norm(n) + 1e-15)
        normals.append(n)
    return np.array(normals)


# Precompute for unit hexagon (side 1, angle 0)
_VERTS_UNIT = _hex_vertices(0.0, 0.0, 1.0, 0.0)          # (6,2)
_NORMALS_UNIT = _edge_normals(1.0, 0.0)                  # (6,2)
# dot_normals[i, j] = _NORMALS_UNIT[i] @ _VERTS_UNIT[j]
_DOT_N = _NORMALS_UNIT @ _VERTS_UNIT.T                   # (6,6)


def _inner_inside_outer(centers, s):
    """
    centers: (12,2). s: outer side length (outer centered at origin, angle 0).
    Returns min clearance over all inner vertices and outer edges.
    Positive means all inside.
    """
    outer_verts = _hex_vertices(0.0, 0.0, s, 0.0)        # (6,2)
    # Precompute outer edge vectors and p1
    p1s = outer_verts
    p2s = np.roll(outer_verts, -1, axis=0)
    edges = p2s - p1s                                     # (6,2)

    min_clear = np.inf
    for cx, cy in centers:
        # inner vertices relative to origin
        iv = _VERTS_UNIT + np.array([cx, cy])            # (6,2)
        # For each outer edge i, clearance = cross(edge_i, v - p1_i)
        # = edge_i[0]*(v[1]-p1_i[1]) - edge_i[1]*(v[0]-p1_i[0])
        # Compute for all 6 vertices and 6 edges: (6,6)
        # cross(edge, v - p1) for each edge i and vertex j
        # = edge_i[0]*(v_j[1]-p1_i[1]) - edge_i[1]*(v_j[0]-p1_i[0])
        # Build matrix: for each edge i, (v - p1_i) for all j
        # (6 edges, 6 vertices, 2 coords)
        diff = iv[None, :, :] - p1s[:, None, :]          # (6,6,2)
        # cross = edges[:,0,:]*(diff[:,:,1]) - edges[:,1,:]*(diff[:,:,0])
        cross = edges[:, 0, None] * diff[:, :, 1] - edges[:, 1, None] * diff[:, :, 0]
        m = np.min(cross)
        if m < min_clear:
            min_clear = m
    return min_clear


def _pairwise_min_gap(centers):
    """Minimum SAT gap among all pairs of inner hexagons (all side 1, angle 0)."""
    n = len(centers)
    # For a hexagon centered at c along unit normal n_i:
    #   max proj = c @ n_i + _DOT_N[i, :].max()
    #   min proj = c @ n_i + _DOT_N[i, :].min()
    col_max = _DOT_N.max(axis=1)   # (6,)
    col_min = _DOT_N.min(axis=1)   # (6,)

    min_gap = np.inf
    for i in range(n):
        c1 = centers[i]
        for j in range(i + 1, n):
            c2 = centers[j]
            # For each of the 6 normals
            proj1_max = c1 @ _NORMALS_UNIT.T + col_max   # (6,)
            proj1_min = c1 @ _NORMALS_UNIT.T + col_min   # (6,)
            proj2_max = c2 @ _NORMALS_UNIT.T + col_max   # (6,)
            proj2_min = c2 @ _NORMALS_UNIT.T + col_min   # (6,)
            # gap along each normal = min(max1,max2) - max(min1,min2)
            gaps = np.minimum(proj1_max, proj2_max) - np.maximum(proj1_min, proj2_min)
            g = np.min(gaps)
            if g < min_gap:
                min_gap = g
    return min_gap


def _objective(x):
    """
    x: [s, cx0, cy0, ..., cx11, cy11]
    Minimize s + penalties for constraint violations.
    """
    s = x[0]
    centers = x[1:].reshape(12, 2)

    cont = _inner_inside_outer(centers, s)
    sep = _pairwise_min_gap(centers)

    pen = 0.0
    if cont < 0:
        pen += 100.0 * (-cont) ** 2
    if sep < 0:
        pen += 100.0 * (-sep) ** 2
    return s + pen


def hexagon_packing_12():
    n = 12

    # Seed: hex-grid cluster of 12 cells
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (1, 1), (-1, 2), (-2, 1), (-2, 0)]
    sq3 = np.sqrt(3.0)
    centers0 = np.array([[1.5 * q, sq3 * (r + q / 2.0)] for q, r in axial])
    centers0 = centers0 - centers0.mean(axis=0)

    # Initial s
    max_r = 0.0
    for c in centers0:
        v = _VERTS_UNIT + c
        max_r = max(max_r, np.max(np.linalg.norm(v, axis=1)))
    s0 = max_r + 0.5

    x0 = np.concatenate([[s0], centers0.flatten()])

    res = minimize(_objective, x0, method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})

    x = res.x
    s = float(x[0])
    centers = x[1:].reshape(12, 2)

    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(s)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
