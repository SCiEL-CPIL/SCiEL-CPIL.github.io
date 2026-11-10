# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_DEADLINE = 600.0
_START = None

def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side s and rotation a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _face_normals(hex_verts):
    """Return 6 outward unit normals for a convex hexagon (CCW vertices)."""
    n = len(hex_verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % n]
        edge = p2 - p1
        # outward normal: rotate edge by -90 deg, normalize
        nrm = np.array([edge[1], -edge[0]])
        nrm = nrm / (np.linalg.norm(nrm) + 1e-15)
        normals[i] = nrm
    return normals

def _pair_separated(v1, n1, v2, n2, tol=1e-6):
    """
    Check if two convex hexagons (each with 6 vertices and 6 normals) are separated.
    Returns True if separated by more than tol on at least one of the 12 normals.
    """
    # For each of the 12 normals, project both polygons and check if intervals are apart
    for i in range(6):
        nrm = n1[i]
        proj1 = v1 @ nrm
        proj2 = v2 @ nrm
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    for i in range(6):
        nrm = n2[i]
        proj1 = v1 @ nrm
        proj2 = v2 @ nrm
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _all_inner_separated(inner_verts, inner_normals, tol=1e-6):
    """Check all pairs of inner hexagons are separated."""
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _pair_separated(inner_verts[i], inner_normals[i], inner_verts[j], inner_normals[j], tol):
                return False
    return True

def _inner_in_outer(inner_verts, outer_verts, tol=1e-6):
    """Check all inner vertices are inside or on the outer hexagon."""
    n_outer = len(outer_verts)
    for iv in inner_verts:
        for p in iv:
            inside = True
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                # cross(p2 - p1, p - p1) should be >= -tol for CCW polygon (inside)
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True

def _edge_lengths_ok(inner_verts, tol=1e-8):
    """Check inner hexagons have edge lengths within tol of 1."""
    for iv in inner_verts:
        for i in range(6):
            p1 = iv[i]
            p2 = iv[(i + 1) % 6]
            length = np.linalg.norm(p2 - p1)
            if abs(length - 1.0) > tol:
                return False
    return True

def _validate(inner_data, outer_data, side):
    """Full validation. Returns (valid, penalty) where penalty is 0 if valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)) or not np.isfinite(side):
        return False, 1e10
    if side < np.sqrt(11.0) - 1e-9:
        return False, 1e10

    # Compute inner hexagon vertices and normals
    inner_verts = []
    inner_normals = []
    for i in range(11):
        cx, cy, a = inner_data[i]
        v = _hex_vertices(cx, cy, 1.0, a)
        inner_verts.append(v)
        inner_normals.append(_face_normals(v))

    # Outer hexagon vertices
    ox, oy, oa = outer_data
    outer_verts = _hex_vertices(ox, oy, side, oa)

    # Shift by outer center
    shift = np.array([ox, oy])
    inner_verts_shifted = [v - shift for v in inner_verts]

    # Check edge lengths
    if not _edge_lengths_ok(inner_verts_shifted):
        return False, 1e10

    # Check separation
    if not _all_inner_separated(inner_verts_shifted, inner_normals):
        return False, 1e10

    # Check containment
    outer_verts_shifted = outer_verts - shift
    if not _inner_in_outer(inner_verts_shifted, outer_verts_shifted):
        return False, 1e10

    return True, 0.0

def _penalty_function(params):
    """
    Objective function for optimization.
    params: [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, side]
    Returns penalty (to minimize). Lower is better.
    """
    global _START
    if _START is None:
        _START = time.time()
    if time.time() - _START > _DEADLINE - 10:
        return 1e10

    inner_data = params[:33].reshape(11, 3)
    side = params[33]

    if side < np.sqrt(11.0):
        return 1e10

    outer_data = np.array([0.0, 0.0, 0.0])

    # Compute vertices
    inner_verts = []
    inner_normals = []
    for i in range(11):
        cx, cy, a = inner_data[i]
        v = _hex_vertices(cx, cy, 1.0, a)
        inner_verts.append(v)
        inner_normals.append(_face_normals(v))

    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)

    # Penalty for separation violations
    sep_penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            # Check all 12 normals
            max_sep = -np.inf
            for k in range(6):
                nrm = inner_normals[i][k]
                proj1 = inner_verts[i] @ nrm
                proj2 = inner_verts[j] @ nrm
                gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                # gap > 0 means separated
                if gap > max_sep:
                    max_sep = gap
            for k in range(6):
                nrm = inner_normals[j][k]
                proj1 = inner_verts[i] @ nrm
                proj2 = inner_verts[j] @ nrm
                gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                if gap > max_sep:
                    max_sep = gap
            if max_sep < 1e-6:
                sep_penalty += max(0.0, 1e-6 - max_sep)

    # Penalty for containment violations
    contain_penalty = 0.0
    n_outer = 6
    for iv in inner_verts:
        for p in iv:
            for i in range(n_outer):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % n_outer]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    contain_penalty += max(0.0, -1e-6 - cross)

    # Edge length penalty
    edge_penalty = 0.0
    for iv in inner_verts:
        for i in range(6):
            p1 = iv[i]
            p2 = iv[(i + 1) % 6]
            length = np.linalg.norm(p2 - p1)
            edge_penalty += (length - 1.0) ** 2

    total = sep_penalty * 1000.0 + contain_penalty * 1000.0 + edge_penalty * 100.0 + side
    return total

def hexagon_packing_11():
    global _START
    _START = time.time()

    # Initial guess: honeycomb arrangement
    # 11 hexagons in a compact arrangement
    # Center + 6 around it + 4 more
    r = 2.0  # distance between centers for unit hexagons (touching)
    init = np.zeros(34)
    # Center
    init[0:3] = [0, 0, 0]
    # 6 surrounding
    for i in range(6):
        a = np.deg2rad(i * 60)
        init[3 + i*3:6 + i*3] = [r * np.cos(a), r * np.sin(a), 0]
    # 4 more: extend the pattern
    # Place at next ring positions
    positions = [
        [2*r * np.cos(np.deg2rad(0)), 2*r * np.sin(np.deg2rad(0))],
        [2*r * np.cos(np.deg2rad(120)), 2*r * np.sin(np.deg2rad(120))],
        [2*r * np.cos(np.deg2rad(240)), 2*r * np.sin(np.deg2rad(240))],
        [r * np.cos(np.deg2rad(30)), r * np.sin(np.deg2rad(30))],
    ]
    for i in range(4):
        idx = 21 + i * 3
        init[idx:idx+3] = [positions[i][0], positions[i][1], 0]
    init[33] = 8.0  # initial side

    # Bounds
    bounds = [(-20, 20)] * 22  # x, y for each hex
    bounds += [(-180, 180)] * 11  # angles
    bounds += [(np.sqrt(11.0), 20.0)]  # side

    # Run differential evolution
    best_result = None
    try:
        result = differential_evolution(
            _penalty_function,
            bounds,
            x0=init,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            update_interval=1,
            callback=lambda x, convergence: None
        )
        if result.success or result.fun < 1e6:
            best_result = result.x
    except Exception:
        pass

    # Try multiple random restarts if time permits
    if best_result is None and time.time() - _START < _DEADLINE - 120:
        for trial in range(5):
            if time.time() - _START > _DEADLINE - 60:
                break
            np.random.seed(trial * 123 + 7)
            init_t = np.random.uniform(-15, 15, size=33)
            init_t[33] = np.random.uniform(np.sqrt(11.0) + 1, 12.0)
            try:
                result_t = differential_evolution(
                    _penalty_function,
                    bounds,
                    x0=init_t,
                    maxiter=100,
                    popsize=12,
                    tol=1e-8,
                    seed=trial * 100 + 1,
                    polish=False
                )
                if result_t.success or result_t.fun < 1e6:
                    if best_result is None or result_t.fun < _penalty_function(best_result):
                        best_result = result_t.x
            except Exception:
                pass

    if best_result is None:
        best_result = init

    inner_data = best_result[:33].reshape(11, 3)
    side = best_result[33]
    outer_data = np.array([0.0, 0.0, 0.0])

    # Validate and if invalid, fall back to a known valid configuration
    valid, pen = _validate(inner_data, outer_data, side)
    if not valid:
        # Fallback: use the seed's known-valid configuration
        inner_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ])
        outer_data = np.array([0, 0, 0])
        side = 8.0

    return inner_data, outer_data, float(side)

# EVOLVE-BLOCK-END
