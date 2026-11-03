"""Combined program: LP-based outer hexagon with bidirectional constraints + smooth warm-start."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side_lp(all_vertices, angle_deg):
    """
    For a fixed outer hexagon angle, find optimal center and side length using LP.
    Uses BIDIRECTIONAL constraints: for each normal n, both n·(v-c) <= r and -n·(v-c) <= r.
    Returns (cx, cy, side) or None on failure.
    """
    n_pts = len(all_vertices)
    if n_pts == 0:
        return 0.0, 0.0, 0.0

    A_ub = []
    b_ub = []

    # Outer hexagon normals for given angle
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            # (vx - cx)*nx + (vy - cy)*ny <= r  =>  -nx*cx - ny*cy - r <= -vx*nx - vy*ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-vx * nx - vy * ny)
            # -(vx - cx)*nx - (vy - cy)*ny <= r  =>  nx*cx + ny*cy - r <= vx*nx + vy*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(vx * nx + vy * ny)

    # r >= 0
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=[(None, None), (None, None), (0, None)],
                         method='highs', options={'presolve': True})
        if result.success:
            cx_opt = result.x[0]
            cy_opt = result.x[1]
            r_opt = result.x[2]
            side = r_opt / (math.sqrt(3) / 2)
            return cx_opt, cy_opt, side
    except Exception:
        pass
    return None

def _logsumexp(x, beta=100.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x = np.array(x)
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _outer_side_smooth(params, all_vertices, outer_angle_deg):
    """
    Smooth objective: minimize the outer hexagon inradius.
    params: [cx, cy, r]
    all_vertices: flat list of (vx, vy)
    """
    cx, cy, r = params

    # For each normal, compute max projection of (v - c) onto n
    # We need: max over all vertices and normals of |n·(v-c)|
    # Smooth: logsumexp of all signed projections (both signs)
    projections = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            proj = (vx - cx) * nx + (vy - cy) * ny
            projections.append(proj)
            projections.append(-proj)

    projections = np.array(projections)
    r_required = _logsumexp(projections, beta=50.0)
    return r_required / (math.sqrt(3) / 2)

_global_all_vertices = []
_global_outer_angle = 0.0

def _objective_smooth(params):
    """Objective function for smooth optimization."""
    if time.time() > _DEADLINE - 2.0:
        return 1e9
    try:
        return _outer_side_smooth(params, _global_all_vertices, _global_outer_angle)
    except Exception:
        return 1e9

def _find_best_outer_for_config(all_vertices):
    """
    Find the best outer hexagon (center, side, angle) for a fixed set of inner vertices.
    Returns (cx, cy, side, angle_deg).
    """
    global _global_all_vertices, _global_outer_angle

    _global_all_vertices = all_vertices

    best_side = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)

    # Centroid for initial guess
    cx0 = sum(v[0] for v in all_vertices) / len(all_vertices)
    cy0 = sum(v[1] for v in all_vertices) / len(all_vertices)

    # Phase 1: Coarse sweep over outer angles (0 to 60, step 2)
    for angle_deg in np.arange(0, 60, 2.0):
        if time.time() > _DEADLINE - 10.0:
            break
        _global_outer_angle = float(angle_deg)

        # LP-based exact evaluation
        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 2: Fine sweep around best angle (step 0.2)
    for angle_deg in np.arange(best_angle - 2.0, best_angle + 2.0, 0.2):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 10.0:
            break
        _global_outer_angle = float(angle_deg)

        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 3: Very fine polish (step 0.02)
    for angle_deg in np.arange(best_angle - 0.2, best_angle + 0.2, 0.02):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 5.0:
            break
        _global_outer_angle = float(angle_deg)

        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    # Phase 4: Final ultra-fine (step 0.002)
    for angle_deg in np.arange(best_angle - 0.02, best_angle + 0.02, 0.002):
        if angle_deg < 0 or angle_deg >= 60:
            continue
        if time.time() > _DEADLINE - 3.0:
            break
        _global_outer_angle = float(angle_deg)

        res = _min_enclosing_hex_side_lp(all_vertices, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = float(angle_deg)
                best_center = (cx, cy)

    return best_center[0], best_center[1], best_side, best_angle

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    sqrt3 = math.sqrt(3)
    scale = 1.000001  # Slightly larger than touching to avoid numerical issues

    # Triangular lattice basis vectors for hexagons of side 1
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    # Generate candidate positions on the triangular lattice
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            dist_sq = x * x + y * y
            candidates.append((dist_sq, x, y))

    candidates.sort(key=lambda t: t[0])

    # Select N closest positions to origin
    positions = []
    for _, x, y in candidates:
        positions.append((x, y))
        if len(positions) == N:
            break

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Try different inner hexagon orientations
    for inner_angle in [0.0, 30.0]:
        if time.time() > _DEADLINE - 10.0:
            break

        # Compute all vertices for this configuration
        all_vertices = []
        for px, py in positions:
            verts = _get_hex_vertices(px, py, inner_angle)
            all_vertices.extend(verts)

        # Find best outer hexagon
        cx, cy, side, angle = _find_best_outer_for_config(all_vertices)

        if side < best_side:
            best_side = side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = (cx, cy)
            best_outer_angle = angle

    if best_inner is None:
        # Fallback if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
