import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar, minimize, differential_evolution

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    A_rows = []
    b_vals = []

    for j in range(6):
        nx, ny = normals[j]
        A_rows.append(np.tile([-nx, -ny, -1.0], (M, 1)))
        b_vals.append(-vdots[:, j])
        A_rows.append(np.tile([nx, ny, -1.0], (M, 1)))
        b_vals.append(vdots[:, j])

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    A = np.vstack([A, np.array([[0.0, 0.0, -1.0]])])
    b = np.concatenate([b, [0.0]])

    c_obj = np.array([0.0, 0.0, 1.0])
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _objective_for_angle(verts, outer_angle_deg):
    """Return the side length for a given outer angle, or a large penalty."""
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]


def _smooth_objective_for_angle(verts, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # shape (M, 6)

    widths = np.max(vdots, axis=0) - np.min(vdots, axis=0)  # shape (6,)
    R_vals = widths / 2.0

    max_R = np.max(R_vals)
    shifted = R_vals - max_R
    log_sum_exp = max_R + np.log(np.sum(np.exp(beta * shifted))) / beta

    side = 2.0 * log_sum_exp * _INVSQRT3
    return side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y, i, j))

    if layout_idx == 0:
        # Compact: center + ring1 (6) + 4 from ring2
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # 3-4-4 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # Staggered 2-4-2-3
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # 2-3-3-3 rows (wider)
        positions = []
        y1 = -2.25 * scale
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -0.75 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.75 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        y4 = 2.25 * scale
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 6:
        # Diamond-like arrangement
        positions = []
        # Center
        positions.append((0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = k * math.pi / 3
            r = sqrt3 * scale
            positions.append((r * math.cos(ang), r * math.sin(ang)))
        # Ring 2: 4 hexagons at 45-degree offsets
        for k in range(4):
            ang = math.pi / 4 + k * math.pi / 2
            r = 2 * sqrt3 * scale
            positions.append((r * math.cos(ang), r * math.sin(ang)))
        return positions[:N]
    else:
        # Fallback: nearest to origin
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _optimize_outer_angle(verts, initial_angle, time_limit):
    """
    Optimize the outer hexagon angle using smooth surrogate + exact refinement.
    Returns (best_angle, best_side).
    """
    best_side = 1e10
    best_angle = initial_angle

    # Step 1: Smooth optimization via L-BFGS-B
    try:
        res_opt = minimize(
            lambda a: _smooth_objective_for_angle(verts, a[0], beta=100.0),
            x0=[initial_angle],
            method='L-BFGS-B',
            bounds=[(0.0, 30.0)],
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        if res_opt.success:
            opt_angle = float(np.clip(res_opt.x[0], 0.0, 30.0))
            if time.time() < time_limit:
                exact_side = _objective_for_angle(verts, opt_angle)
                if exact_side < best_side:
                    best_side = exact_side
                    best_angle = opt_angle
    except Exception:
        pass

    # Step 2: Bounded scalar minimization with exact objective
    lo = max(0.0, best_angle - 3.0)
    hi = min(30.0, best_angle + 3.0)
    try:
        res_scalar = minimize_scalar(
            lambda a: _objective_for_angle(verts, a),
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 200}
        )
        if res_scalar.success:
            if res_scalar.fun < best_side:
                best_side = res_scalar.fun
                best_angle = float(res_scalar.x)
    except Exception:
        pass

    # Step 3: Fine grid polish around best angle
    if time.time() < time_limit:
        for delta_max, step in [(0.005, 0.0005), (0.001, 0.00005), (0.0002, 0.00001)]:
            if time.time() > time_limit:
                break
            lo2 = max(0.0, best_angle - delta_max)
            hi2 = min(30.0, best_angle + delta_max)
            n_steps = int((hi2 - lo2) / step) + 1
            if n_steps > 2000:
                step = (hi2 - lo2) / 2000
            for a in np.arange(lo2, hi2 + step / 2, step):
                if time.time() > time_limit:
                    break
                s = _objective_for_angle(verts, a)
                if s < best_side:
                    best_side = s
                    best_angle = float(a)

    return best_angle, best_side


def _sdf_hexagon_distance(px, py, hx, hy, angle_deg):
    """
    Compute signed distance from point (px, py) to a unit hexagon centered at (hx, hy)
    with rotation angle_deg. Positive means outside, negative means inside.
    For disjointness, we need distance >= 0 for all vertex-to-hexagon pairs.
    """
    a0 = math.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = px - hx
    dy = py - hy
    cos_a = math.cos(-a0)
    sin_a = math.sin(-a0)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    
    # For a regular hexagon with side length 1, the inradius is sqrt(3)/2
    # The hexagon in local frame has vertices at angles 0, 60, 120, 180, 240, 300
    # Each face is at distance sqrt(3)/2 from center
    # The face normals are at angles 30, 90, 150, 210, 270, 330 degrees
    inradius = _SQRT3 / 2.0
    
    # Compute distance to each of the 6 faces
    min_dist = float('inf')
    for k in range(6):
        face_angle = math.radians(30 + k * 60)
        nx = math.cos(face_angle)
        ny = math.sin(face_angle)
        # Signed distance to this face (positive means outside)
        d = lx * nx + ly * ny - inradius
        if d < min_dist:
            min_dist = d
    
    return min_dist


def _compute_pairwise_min_distance(positions, angles):
    """
    For each pair of hexagons, compute the minimum distance between them.
    Returns the minimum over all pairs (should be >= 0 for disjointness).
    """
    n = len(positions)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ai = positions[i]
            xj, yj, aj = positions[j]
            
            # Check all vertices of hex i against hex j
            vi = _hex_vertices(xi, yi, ai)
            for vx, vy in vi:
                d = _sdf_hexagon_distance(vx, vy, xj, yj, aj)
                if d < min_dist:
                    min_dist = d
            
            # Check all vertices of hex j against hex i
            vj = _hex_vertices(xj, yj, aj)
            for vx, vy in vj:
                d = _sdf_hexagon_distance(vx, vy, xi, yi, ai)
                if d < min_dist:
                    min_dist = d
    
    return min_dist


def _optimize_positions_sdf(initial_positions, initial_angles, time_limit):
    """
    Use SDF-based gradient descent to optimize positions and angles.
    Objective: minimize the maximum outer hexagon side length,
    subject to all pairwise distances >= 0.
    """
    n = len(initial_positions)
    
    # Flatten parameters: [x0, y0, a0, x1, y1, a1, ..., x_{n-1}, y_{n-1}, a_{n-1}]
    params = np.array([p[0] for p in initial_positions] + 
                      [p[1] for p in initial_positions] + 
                      list(initial_angles))
    
    def _objective(params):
        if time.time() > time_limit:
            return 1e10
        
        positions = [(params[3*i], params[3*i+1]) for i in range(n)]
        angles = [params[3*i+2] * 180.0 / math.pi for i in range(n)]
        
        # Check disjointness
        min_dist = _compute_pairwise_min_distance(positions, angles)
        if min_dist < -1e-6:
            # Penalize overlap
            return 1e6 + (min_dist + 1e-6) ** 2
        
        # Compute all vertices
        all_vertices = np.vstack(
            [_hex_vertices(px, py, ang) for px, py, ang in zip(positions, angles)]
        )
        
        # Find best outer hexagon
        best_side = 1e10
        for outer_angle in np.arange(0, 30.25, 1.0):
            res = _min_enclosing_hex_side(all_vertices, outer_angle)
            if res:
                if res[2] < best_side:
                    best_side = res[2]
        
        return best_side
    
    # Use differential evolution for global search
    bounds = [(-5, 5)] * (3 * n)
    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            x0=params,
            maxiter=50,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=False,
            time_limit=max(1, int(time_limit - time.time()))
        )
        if result.fun < 1e5:
            return result.x
    except Exception:
        pass
    
    return params


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 7
    layouts = [_build_positions(i) for i in range(num_layouts)]

    # Inner angles to try
    inner_angles = [0.0, 30.0, 15.0, -15.0, 7.5, -7.5]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break

        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 8.0:
                break

            all_vertices = np.vstack(
                [_hex_vertices(px, py, inner_angle) for px, py in positions]
            )

            # Coarse grid to find good starting angles for outer hex
            coarse_best = 1e10
            coarse_best_angle = 0.0
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 1.0):
                if time.time() > _DEADLINE - 8.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
                    if res[2] < coarse_best:
                        coarse_best = res[2]
                        coarse_best_angle = outer_angle_deg

            if not coarse_results:
                continue

            # Try optimization from top-3 coarse starting points
            coarse_results.sort(key=lambda x: x[0])
            candidate_starts = [coarse_results[0][1]]
            for s, a in coarse_results[1:3]:
                if abs(a - candidate_starts[0]) > 0.5:
                    candidate_starts.append(a)

            local_best_side = coarse_best
            local_best_angle = coarse_best_angle

            for start_a in candidate_starts:
                if time.time() > _DEADLINE - 12.0:
                    break
                ang, side = _optimize_outer_angle(all_vertices, start_a, _DEADLINE - 15.0)
                if side < local_best_side:
                    local_best_side = side
                    local_best_angle = ang

            if local_best_side < best_side:
                res_final = _min_enclosing_hex_side(all_vertices, local_best_angle)
                if res_final:
                    cx, cy, side = res_final
                    if side < best_side:
                        best_side = side
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        best_center = (cx, cy)
                        best_outer_angle = local_best_angle

    # Additional SDF-based optimization if time permits
    if time.time() < _DEADLINE - 20.0 and best_inner is not None:
        # Try to improve the best layout found so far using SDF optimization
        best_positions = [(p[0], p[1]) for p in best_inner]
        best_angle = best_inner[0][2]
        
        # Run SDF optimization
        time_remaining = _DEADLINE - time.time() - 10.0
        if time_remaining > 10.0:
            try:
                optimized_params = _optimize_positions_sdf(
                    best_positions, [best_angle] * N, time.time() + time_remaining
                )
                if optimized_params is not None:
                    opt_positions = [(optimized_params[3*i], optimized_params[3*i+1]) for i in range(N)]
                    opt_angles = [optimized_params[3*i+2] * 180.0 / math.pi for i in range(N)]
                    
                    # Verify disjointness
                    min_dist = _compute_pairwise_min_distance(opt_positions, opt_angles)
                    if min_dist >= -1e-6:
                        all_vertices = np.vstack(
                            [_hex_vertices(px, py, ang) for px, py, ang in zip(opt_positions, opt_angles)]
                        )
                        
                        # Find best outer hexagon
                        opt_best_side = 1e10
                        opt_best_angle = 0.0
                        for outer_angle in np.arange(0, 30.25, 0.5):
                            res = _min_enclosing_hex_side(all_vertices, outer_angle)
                            if res and res[2] < opt_best_side:
                                opt_best_side = res[2]
                                opt_best_angle = outer_angle
                        
                        if opt_best_side < best_side:
                            res_final = _min_enclosing_hex_side(all_vertices, opt_best_angle)
                            if res_final:
                                cx, cy, side = res_final
                                if side < best_side:
                                    best_side = side
                                    best_inner = [[px, py, ang] for px, py, ang in zip(opt_positions, opt_angles)]
                                    best_center = (cx, cy)
                                    best_outer_angle = opt_best_angle
            except Exception:
                pass

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
