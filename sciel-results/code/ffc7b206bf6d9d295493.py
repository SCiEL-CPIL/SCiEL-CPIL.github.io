# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _lattice_positions():
    """
    12 unit-hexagon centers on a hexagonal lattice, non-overlapping by construction.
    Using axial coordinates (q, r) on a triangular lattice with spacing sqrt(3),
    adjacent hexagons (distance sqrt(3) apart) are exactly touching (non-overlapping).
    """
    s3 = np.sqrt(3.0)
    # Axial to Cartesian: x = s3*(q + r/2), y = 1.5*r
    axials = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),  # ring 1 (6)
        (2, 0), (2, -1), (1, -2), (-1, 1), (-1, 2),          # 5 more to make 12
    ]
    # Wait, let me use a cleaner set: center + 6 ring1 + 5 from ring2
    axials = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, 2), (-1, 1),
    ]
    # Hmm, (-1,1) is already in ring1. Let me pick 5 distinct ring-2 positions.
    # Ring 2 axial coords (|q|<=2, |r|<=2, |q+r|<=2, and max(|q|,|r|,|q+r|)==2):
    # (2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-2),(-2,-2),(-2,-1),(-2,0),(-1,1),(0,2),(1,2)
    # Pick 5 that are spread: (2,0), (1,-2), (-1,-2), (-2,0), (-1,1)
    # But (-1,1) is ring1. Let me use: (2,0), (2,-1), (1,-2), (-1,-2), (-2,0)
    axials = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (-1, -2), (-2, 0),
    ]
    positions = np.zeros((12, 2))
    for i, (q, r) in enumerate(axials):
        positions[i, 0] = s3 * (q + r / 2.0)
        positions[i, 1] = 1.5 * r
    return positions


def _min_containing_hex_side(centers):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0), angle 0,
    that contains all unit hexagons with given centers.
    """
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    verts_flat = verts.reshape(-1, 2)
    projections = normals @ verts_flat.T
    max_proj_per_normal = np.max(projections, axis=1)
    max_proj = np.max(max_proj_per_normal)

    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _check_separation(centers, tol=1e-6):
    """Check if all pairs of unit hexagons are separated using SAT."""
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]

            proj_i = vi @ normals.T
            proj_j = vj @ normals.T

            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)

            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _objective(params):
    """
    params: [scale, angle_rad, tx, ty]
    Apply affine map to the fixed lattice positions, compute containing hex side.
    Add penalty if separation is violated (shouldn't happen if scale is reasonable).
    """
    scale, angle, tx, ty = params
    if scale <= 0:
        return 1e10

    base_pos = _lattice_positions()
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    centers = base_pos * scale @ R.T + np.array([tx, ty])

    s_contain = _min_containing_hex_side(centers)

    # Separation check: at scale=1, lattice is exactly touching.
    # At scale > 1, they separate. At scale < 1, they overlap.
    # We need scale >= 1 for non-overlap.
    # Actually, the lattice positions are for unit hexagons at scale=1 (touching).
    # If we scale positions by 'scale', the hexagons themselves are still unit,
    # but their centers move apart by factor 'scale'.
    # So separation is guaranteed iff scale >= 1.
    # But wait - the hexagons are always unit (side 1). Only centers move.
    # At scale=1, adjacent centers are sqrt(3) apart -> touching.
    # At scale>1, they're further apart -> separated.
    # At scale<1, they overlap.
    # So we need scale >= 1.

    if scale < 1.0 - 1e-9:
        return s_contain + 100.0 * (1.0 - scale) ** 2

    return s_contain


def _affine_optimize(init_params):
    """Run Nelder-Mead on the 4-parameter affine map."""
    try:
        result = minimize(
            _objective,
            init_params,
            method='Nelder-Mead',
            options={
                'maxiter': 10000,
                'maxfev': 100000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
        )
        return result.x
    except Exception:
        return init_params


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    base_pos = _lattice_positions()
    best_centers = None
    best_s = np.inf
    best_params = None

    # Initial parameter guesses: scale=1 (touching), various rotations, centered at origin
    rng = np.random.default_rng(42)

    # Try a range of rotation angles and scales
    init_params_list = []
    for angle_deg in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        for scale in [1.0, 1.05, 1.1, 1.15, 1.2]:
            angle_rad = np.deg2rad(angle_deg)
            # Center the lattice at origin (it already is, approximately)
            centroid = np.mean(base_pos, axis=0)
            tx = -centroid[0] * scale
            ty = -centroid[1] * scale
            init_params_list.append(np.array([scale, angle_rad, tx, ty]))

    # Also add some random initializations
    for _ in range(20):
        angle_rad = rng.uniform(0, 2 * np.pi)
        scale = rng.uniform(1.0, 1.3)
        tx = rng.uniform(-3, 3)
        ty = rng.uniform(-3, 3)
        init_params_list.append(np.array([scale, angle_rad, tx, ty]))

    for init_params in init_params_list:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break

        optimized_params = _affine_optimize(init_params)

        # Extract centers
        scale, angle, tx, ty = optimized_params
        if scale < 1.0 - 1e-6:
            continue

        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        centers = base_pos * scale @ R.T + np.array([tx, ty])

        # Verify separation
        if not _check_separation(centers):
            continue

        s_contain = _min_containing_hex_side(centers)
        if s_contain < best_s:
            best_s = s_contain
            best_centers = centers.copy()
            best_params = optimized_params.copy()

    # Fallback: use the lattice at scale=1, rotation=0, centered
    if best_centers is None:
        centroid = np.mean(base_pos, axis=0)
        best_centers = base_pos - centroid
        best_s = _min_containing_hex_side(best_centers)

    # Final polish: one more Nelder-Mead from the best found
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - _SAFETY_MARGIN and best_params is not None:
        refined_params = _affine_optimize(best_params)
        scale, angle, tx, ty = refined_params
        if scale >= 1.0 - 1e-6:
            c = np.cos(angle)
            s = np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            centers = base_pos * scale @ R.T + np.array([tx, ty])
            if _check_separation(centers):
                s_contain = _min_containing_hex_side(centers)
                if s_contain < best_s:
                    best_s = s_contain
                    best_centers = centers.copy()

    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = best_s

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
