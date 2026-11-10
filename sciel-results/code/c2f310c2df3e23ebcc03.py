# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return (6,2) array of vertices for hexagon centered at (cx,cy) with side s, angle a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _separation_ok(v1, v2, gap=1e-6):
    """Check if two hexagons (each 6 vertices) are separated.
    Returns True if on at least one of 12 edge normals the projected intervals are apart by > gap."""
    # 12 edge normals at angles k*30 degrees
    normals = np.array([np.array([np.cos(np.deg2rad(k * 30)), np.sin(np.deg2rad(k * 30))])
                        for k in range(12)])
    # Project vertices onto each normal
    p1 = v1 @ normals.T  # (6,12)
    p2 = v2 @ normals.T  # (6,12)
    for k in range(12):
        min1, max1 = np.min(p1[:, k]), np.max(p1[:, k])
        min2, max2 = np.min(p2[:, k]), np.max(p2[:, k])
        # Intervals [min1,max1] and [min2,max2] are apart if max1 < min2 or max2 < min1
        if max1 < min2 - gap or max2 < min1 - gap:
            return True
    return False


def _inside_outer(v, outer_v, tol=-1e-6):
    """Check if all points v (n,2) are inside or on outer hexagon with vertices outer_v (6,2)."""
    for i in range(6):
        p1 = outer_v[i]
        p2 = outer_v[(i + 1) % 6]
        edge = p2 - p1
        # cross(p2-p1, p-p1) for all points
        rel = v - p1
        cross = rel[:, 0] * edge[1] - rel[:, 1] * edge[0]
        if np.any(cross < tol):
            return False
    return True


def _is_valid(inner_xy, side, outer_center=np.array([0.0, 0.0])):
    """Full validity check for a configuration."""
    if not np.all(np.isfinite(inner_xy)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    # Check all pairs of inner hexagons are separated
    verts = [_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy]
    for i in range(11):
        for j in range(i + 1, 11):
            if not _separation_ok(verts[i], verts[j]):
                return False
    # Check all inner vertices are inside outer hexagon
    outer_v = _hex_vertices(outer_center[0], outer_center[1], side, 0.0)
    for v in verts:
        if not _inside_outer(v, outer_v):
            return False
    # Check edge lengths of inner hexagons (should be 1)
    for v in verts:
        for i in range(6):
            d = np.linalg.norm(v[(i+1)%6] - v[i])
            if abs(d - 1.0) > 1e-8:
                return False
    return True


def _separation_penalty(inner_xy, gap=1e-6):
    """Compute penalty for overlapping hexagons. Returns 0 if all separated."""
    verts = [_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy]
    normals = np.array([np.array([np.cos(np.deg2rad(k * 30)), np.sin(np.deg2rad(k * 30))])
                        for k in range(12)])
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            p1 = verts[i] @ normals.T  # (6,12)
            p2 = verts[j] @ normals.T  # (6,12)
            separated = False
            for k in range(12):
                min1, max1 = np.min(p1[:, k]), np.max(p1[:, k])
                min2, max2 = np.min(p2[:, k]), np.max(p2[:, k])
                # Intervals are apart if max1 < min2 - gap or max2 < min1 - gap
                # Penalty if they overlap (not separated)
                if max1 < min2 - gap or max2 < min1 - gap:
                    separated = True
                    break
            if not separated:
                # Compute overlap amount as penalty
                for k in range(12):
                    min1, max1 = np.min(p1[:, k]), np.max(p1[:, k])
                    min2, max2 = np.min(p2[:, k]), np.max(p2[:, k])
                    # Overlap if max1 > min2 + gap and max2 > min1 + gap
                    overlap = max(0, max1 - (min2 + gap)) + max(0, max2 - (min1 + gap))
                    penalty += overlap
    return penalty


def _min_side_for_positions(inner_xy):
    """Find minimum side of outer hex (centered at origin, angle 0) containing all inner hexes."""
    normals = np.array([[np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))]
                       for k in range(6)])
    all_verts = np.vstack([_hex_vertices(x, y, 1.0, 0.0) for x, y in inner_xy])
    projections = all_verts @ normals.T  # (66, 6)
    max_proj = np.max(projections, axis=1)  # (66,)
    s_min = np.max(max_proj) / (np.sqrt(3) / 2)
    return s_min


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    t_start = time.time()
    deadline = 620.0  # safety margin before 630s limit

    best_side = np.inf
    best_inner = None
    s3 = np.sqrt(3)

    # Start with a well-separated initial configuration
    # Flower pattern with proper spacing to avoid overlaps
    # For unit hexagons with angle 0, neighbors in honeycomb are at distance 2.
    # Scale by 1.01 to ensure gap > 1e-6 (touching is intersection)
    scale = 1.01
    init_xy = np.array([
        [0, 0],
        [2 * scale, 0],
        [-2 * scale, 0],
        [1 * scale, s3 * scale],
        [-1 * scale, s3 * scale],
        [1 * scale, -s3 * scale],
        [-1 * scale, -s3 * scale],
        [3 * scale, s3 * scale],
        [3 * scale, -s3 * scale],
        [-3 * scale, s3 * scale],
        [-3 * scale, -s3 * scale],
    ])

    # Objective: minimize side length + penalty for overlaps
    def objective(params):
        if time.time() > deadline:
            return 1e10
        xy = params.reshape(11, 2)
        if not np.all(np.isfinite(xy)):
            return 1e10
        side = _min_side_for_positions(xy)
        sep_pen = _separation_penalty(xy)
        # Use a stronger penalty weight to prioritize feasibility
        return side + 50.0 * sep_pen

    # Optimize with Nelder-Mead
    try:
        res = minimize(objective, init_xy.flatten(), method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-11,
                               'adaptive': True, 'maxfev': 20000})
        opt_xy = res.x.reshape(11, 2)
        opt_side = _min_side_for_positions(opt_xy)
        if _is_valid(opt_xy, opt_side):
            best_side = opt_side
            best_inner = opt_xy
    except Exception:
        pass

    # Try additional local optimizations from different starting points
    if time.time() < deadline - 60:
        # Perturbed starts with smaller perturbations to stay near feasible region
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            perturbed = init_xy + rng.normal(0, 0.05, size=(11, 2))
            try:
                res2 = minimize(objective, perturbed.flatten(), method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11,
                                       'adaptive': True, 'maxfev': 10000})
                opt_xy2 = res2.x.reshape(11, 2)
                opt_side2 = _min_side_for_positions(opt_xy2)
                if _is_valid(opt_xy2, opt_side2):
                    if opt_side2 < best_side:
                        best_side = opt_side2
                        best_inner = opt_xy2
            except Exception:
                pass

    # If we found a valid configuration, return it
    if best_inner is not None:
        inner_hex_data = np.column_stack([best_inner, np.zeros(11)])
        outer_hex_data = np.array([0.0, 0.0, 0.0])
        return inner_hex_data, outer_hex_data, float(best_side)

    # Fallback: return the initial configuration (which is valid due to scaling)
    side = _min_side_for_positions(init_xy)
    # Ensure it's valid; if not, increase side
    if not _is_valid(init_xy, side):
        # Try increasing side until valid (shouldn't happen with scale=1.01)
        side = 15.0
    inner_hex_data = np.column_stack([init_xy, np.zeros(11)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
