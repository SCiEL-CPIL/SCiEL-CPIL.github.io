# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _cluster_vertices(params):
    """params: 33 floats = 11*(x, y, angle_deg). Returns (11,6,2) vertices."""
    verts = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, 1.0, ad)
    return verts


def _pair_gap(verts_a, verts_b):
    """Minimum separation between two unit hexagons (positive if disjoint, negative if overlap).
    We approximate by checking 12 edge normals of hexagon A and 12 of hexagon B.
    For each normal, project both sets of vertices; if the projected intervals are disjoint
    (gap > 0), that normal proves separation. The hexagons are disjoint iff at least one
    of the 24 normals (or equivalently 12 of A's, since convex) gives a positive gap.
    Return the max gap over all 12 normals of A (if >0, disjoint; if <=0 for all, overlapping)."""
    # Edge normals of a unit regular hexagon: directions perpendicular to each edge.
    # For a hexagon with angle a, edge i connects vertex i to vertex i+1.
    # The outward normal is the direction from center to midpoint of edge i.
    # Simpler: use the 6 face normals of a regular hexagon (pointing to edge midpoints).
    # For unit hexagon at origin with angle 0, vertices at angles 0,60,...,300.
    # Edge midpoints at angles 30,90,150,210,270,330. Normals = (cos, sin) of those.
    def normals(a_deg):
        a = np.deg2rad(a_deg)
        mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
        return np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])

    # Get centers and angles
    # Re-extract from verts: center is mean of vertices
    center_a = verts_a.mean(axis=0)
    center_b = verts_b.mean(axis=0)
    # angle: from first vertex
    a_a = np.rad2deg(np.arctan2(verts_a[0, 1] - center_a[1], verts_a[0, 0] - center_a[0]))
    a_b = np.rad2deg(np.arctan2(verts_b[0, 1] - center_b[1], verts_b[0, 0] - center_b[0]))

    n_a = normals(a_a)
    n_b = normals(a_b)

    best_gap = -np.inf
    for n in np.vstack([n_a, n_b]):
        proj_a = verts_a @ n
        proj_b = verts_b @ n
        gap = max(proj_b.min() - proj_a.max(), proj_a.min() - proj_b.max())
        if gap > best_gap:
            best_gap = gap
    return best_gap


def _objective(params):
    """Objective to minimize: a penalty if any pair overlaps, plus the required outer side length."""
    if not np.all(np.isfinite(params)):
        return 1e10
    verts = _cluster_vertices(params)

    # Check pairwise separation
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j])
            if g < min_gap:
                min_gap = g
    if min_gap < 1e-6:
        # Overlap: heavy penalty proportional to how deep
        return 1e6 * (1e-6 - min_gap) + 100.0

    # Compute minimal enclosing regular hexagon side length
    # For a set of points, the minimal side of a regular hexagon (any orientation) that contains them.
    # The inradius of a regular hexagon with side s is s * sqrt(3)/2.
    # A point (x,y) is inside a hexagon centered at origin with inradius r and orientation a iff
    #   max over 6 face normals n_k of (n_k . (x,y)) <= r
    # where face normals are at angles a + 30 + 60k.
    # So for a given orientation a, r(a) = max over all points p and all 6 normals n_k of (n_k . p).
    # We need r(a) for the best orientation a, then side = r(a) / (sqrt(3)/2) = 2*r(a)/sqrt(3).
    # Also we can translate the hexagon center to the centroid to minimize r.
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)
    shifted = all_verts - centroid

    # Optimize orientation a in [0, 60)
    best_side = np.inf
    for a_deg in np.linspace(0, 60, 61):
        a = np.deg2rad(a_deg)
        mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])  # (6,2)
        # For each point, max over normals of dot
        # shifted: (66,2), normals: (6,2)
        dots = shifted @ normals.T  # (66,6)
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side

    return best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    using rigid-body cluster optimization. Returns (inner, outer, side).
    """
    # Initial guess: a compact hexagonal cluster arrangement
    # Use a flower-like arrangement: 1 center + 6 around + 4 more
    init = []
    # center
    init.append([0.0, 0.0, 0.0])
    # 6 around at distance ~2 (touching: center-to-center for touching unit hexagons is 2)
    for k in range(6):
        ang = np.deg2rad(60 * k)
        init.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    # 4 more: place at second ring positions
    for k in [0, 2, 4, 5]:  # pick 4 of 6 second-ring positions
        ang = np.deg2rad(60 * k + 30)
        init.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])

    init_params = np.array(init).flatten()

    # Bounds for differential evolution
    # Positions: allow spread within [-10, 10], angles in [-180, 180]
    bounds = []
    for i in range(11):
        bounds.append((-8.0, 8.0))   # x
        bounds.append((-8.0, 8.0))   # y
        bounds.append((-180.0, 180.0))  # angle

    result = differential_evolution(
        _objective,
        bounds,
        x0=init_params,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        seed=42,
        polish=False,
        init='sobol',
        workers=-1,
    )

    best_params = result.x
    if not np.all(np.isfinite(best_params)):
        best_params = init_params

    # Build inner hex data
    inner_hex_data = np.empty((11, 3))
    for i in range(11):
        inner_hex_data[i] = best_params[3 * i:3 * i + 3]

    # Compute outer hexagon: centroid and optimal orientation
    verts = _cluster_vertices(best_params)
    all_verts = verts.reshape(-1, 2)
    centroid = all_verts.mean(axis=0)

    # Find best orientation for minimal side
    shifted = all_verts - centroid
    best_side = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0, 60, 121):
        a = np.deg2rad(a_deg)
        mid_angles = a + np.deg2rad(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(mid_angles), np.sin(mid_angles)])
        dots = shifted @ normals.T
        r = dots.max()
        side = 2.0 * r / np.sqrt(3.0)
        if side < best_side:
            best_side = side
            best_a = a_deg

    # Shift inner hexagons so that the outer hexagon is centered at origin
    # The outer hexagon center is at centroid. We shift everything so centroid -> origin.
    inner_hex_data[:, 0] -= centroid[0]
    inner_hex_data[:, 1] -= centroid[1]

    outer_hex_data = np.array([0.0, 0.0, best_a])
    outer_hex_side_length = float(best_side)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
