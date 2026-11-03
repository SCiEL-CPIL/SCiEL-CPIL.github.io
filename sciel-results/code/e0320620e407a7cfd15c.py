"""Combined program: LP-based outer hexagon + multi-layout search + polish."""
import math
import time
import numpy as np
import itertools

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices_array(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg.
    Returns numpy array shape (6, 2)."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex direction at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    Uses LP with support function: only need max/min of (v-c).n over all v.
    For a convex polygon, these occur at vertices. We precompute per-normal
    max and min to reduce LP size.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)

    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T  # (M, 6)

    # For each normal k, the constraint is:
    #   max_v (v.n_k) - (c.n_k) <= R   =>  (c.n_k) + R >= max_v(v.n_k)
    #   (c.n_k) - min_v(v.n_k) <= R    =>  (c.n_k) - R <= min_v(v.n_k)
    # i.e.  c.n_k - R <= min_k  and  c.n_k + R >= max_k
    # Rewrite as:
    #   nx*cx + ny*cy - R <= min_k
    #  -nx*cx - ny*cy + R >= max_k  =>  nx*cx + ny*cy - R <= -max_k  (wrong sign)
    # Let me redo:
    #   (v - c).n <= R for all v  =>  max_v(v.n) - c.n <= R  =>  c.n + R >= max_v(v.n)
    #   (c - v).n <= R for all v  =>  c.n - min_v(v.n) <= R  =>  c.n - R <= min_v(v.n)
    #
    # LP form (A_ub @ x <= b_ub):
    #   c.n + R >= max_k  =>  -(nx)*cx -(ny)*cy - R <= -max_k
    #   c.n - R <= min_k  =>   nx*cx + ny*cy - R <= min_k

    max_dots = np.max(v_dot_n, axis=0)  # (6,)
    min_dots = np.min(v_dot_n, axis=0)  # (6,)

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        # Constraint 1: -nx*cx - ny*cy - R <= -max_dots[k]
        A_rows.append([-nx, -ny, -1.0])
        b_vals.append(-max_dots[k])
        # Constraint 2: nx*cx + ny*cy - R <= min_dots[k]
        A_rows.append([nx, ny, -1.0])
        b_vals.append(min_dots[k])

    A = np.array(A_rows)
    b = np.array(b_vals)

    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _min_enclosing_hex_side_smooth(verts, outer_angle_deg, beta=50.0):
    """
    Smooth approximation of the min enclosing hexagon side using log-sum-exp.
    Returns (cx, cy, side) or None.
    """
    M = len(verts)
    if M == 0:
        return 0.0, 0.0, 0.0

    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    v_dot_n = verts @ normals.T  # (M, 6)

    def objective(params):
        cx, cy, R = params
        if R < 0:
            R = 0.0
        # Constraints: for each vertex v and normal k: | (v-c).n_k | <= R
        # Soft constraint: sum exp(beta * max(0, |(v-c).n_k| - R))
        diffs = v_dot_n - (cx * normals[0, 0] + cy * normals[0, 1])  # wrong, need per-normal
        # Actually: (v - c) . n_k = v.n_k - (cx*n_kx + cy*n_ky)
        c_dot_n = np.array([cx * normals[k, 0] + cy * normals[k, 1] for k in range(6)])
        # diffs[m, k] = v_dot_n[m, k] - c_dot_n[k]
        diffs = v_dot_n - c_dot_n[np.newaxis, :]
        violations = np.abs(diffs) - R
        violations = np.maximum(violations, 0.0)
        # Log-sum-exp for numerical stability
        max_v = np.max(violations)
        if max_v > 50.0:
            return max_v * 10.0 + 1e10
        log_sum = np.log(np.sum(np.exp(beta * violations))) + beta * max_v
        return log_sum / beta + R * 0.01  # slight penalty on R

    # Initial guess: center at mean, R at max distance
    cx0 = np.mean(verts[:, 0])
    cy0 = np.mean(verts[:, 1])
    dists = np.sqrt((verts[:, 0] - cx0)**2 + (verts[:, 1] - cy0)**2)
    R0 = np.max(dists) * 1.2

    from scipy.optimize import minimize
    try:
        res = minimize(objective, [cx0, cy0, R0], method='L-BFGS-B',
                       bounds=[(None, None), (None, None), (0, None)],
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            cx, cy, R = res.x
            if R < 0:
                R = 0.0
            side = R / (_SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _search_outer_angle(verts, coarse_step=0.5, fine_step=0.1, finer_step=0.02, finest_step=0.005):
    """Multi-resolution angle sweep for the outer hexagon.
    Returns (cx, cy, side, angle_deg)."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse: 0 to 60 in steps of coarse_step
    for angle_deg in np.arange(0, 60, coarse_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_side == float('inf'):
        return 0.0, 0.0, float('inf'), 0.0

    # Fine
    for angle_deg in np.arange(max(0.0, best_angle - 2.0), min(60.0, best_angle + 2.0), fine_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Finer
    for angle_deg in np.arange(max(0.0, best_angle - 0.5), min(60.0, best_angle + 0.5), finer_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Finest
    for angle_deg in np.arange(max(0.0, best_angle - 0.15), min(60.0, best_angle + 0.15), finest_step):
        if time.time() > _DEADLINE - 10.0:
            break
        res = _min_enclosing_hex_side_lp(verts, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle

def _polish_with_nelder_mead(verts, cx0, cy0, ang0):
    """Polish the outer angle using Nelder-Mead on a 1D problem (angle only).
    The center is always LP-optimal for a given angle, so we just optimize angle.
    Returns improved (cx, cy, side, angle) or None."""
    if time.time() > _DEADLINE - 15.0:
        return None

    from scipy.optimize import minimize

    def objective(params):
        ang_p = params[0] % 60.0
        res = _min_enclosing_hex_side_lp(verts, float(ang_p))
        if res is None:
            return 1e10
        return res[2]

    x0 = np.array([ang0])
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14,
                                'adaptive': True})
        if res.success:
            ang_nm = float(res.x[0] % 60.0)
            res2 = _min_enclosing_hex_side_lp(verts, ang_nm)
            if res2 is not None:
                return res2[0], res2[1], res2[2], ang_nm
    except Exception:
        pass

    # Also try a local golden-section search for very fine angle tuning
    try:
        a, b = max(0.0, ang0 - 0.05), min(60.0, ang0 + 0.05)
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = objective([c])
        fd = objective([d])
        for _ in range(80):
            if time.time() > _DEADLINE - 10.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = objective([c])
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = objective([d])
        ang_best = (a + b) / 2.0
        ang_best = float(ang_best % 60.0)
        res2 = _min_enclosing_hex_side_lp(verts, ang_best)
        if res2 is not None:
            return res2[0], res2[1], res2[2], ang_best
    except Exception:
        pass

    return None

def _build_layouts():
    """Generate multiple candidate layouts for the 11 inner hexagons.
    Returns list of (positions_list, inner_angle)."""
    layouts = []
    scale = 1.000001
    u = (_SQRT3 * scale, 0.0)
    v = (_SQRT3 / 2 * scale, 1.5 * scale)

    # Layout 0: hexagonal flower + 4 from second ring (compact)
    target = [(0, 0)]
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    target.extend(ring1)
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
    target.extend(ring2_sorted[:4])

    for inner_angle in [0.0, 30.0]:
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        layouts.append((positions[:N], inner_angle))

    # Layout 1: row-based 3-4-4
    positions = []
    y1 = -1.5 * scale
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 3):
        x = k * _SQRT3 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-1, 3):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 2: 4-3-4 shifted
    positions = []
    y1 = -1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y1))
    y2 = 0.0
    for k in range(-1, 2):
        x = k * _SQRT3 * scale + _SQRT3 / 2 * scale
        positions.append((x, y2))
    y3 = 1.5 * scale
    for k in range(-2, 2):
        x = k * _SQRT3 * scale
        positions.append((x, y3))
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions[:N], inner_angle))

    # Layout 3: greedy nearest-neighbor from origin on triangular lattice
    coords = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            coords.append((x, y))
    coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
    positions = coords_sorted[:N]
    for inner_angle in [0.0, 30.0]:
        layouts.append((positions, inner_angle))

    # Layout 4: slightly different second-ring selection (pick 4 that minimize bounding)
    # Try different 4-of-10 combinations from ring2
    best_combo_side = float('inf')
    best_combo = None
    for combo in itertools.combinations(ring2, 4):
        target_test = [(0, 0)] + ring1 + list(combo)
        positions_test = []
        for ti, tj in target_test:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions_test.append((x, y))
        # Quick bounding check: use axis-aligned bounding box as proxy
        xs = [p[0] for p in positions_test]
        ys = [p[1] for p in positions_test]
        bbox = (max(xs) - min(xs)) + (max(ys) - min(ys))
        if bbox < best_combo_side:
            best_combo_side = bbox
            best_combo = combo

    if best_combo:
        target = [(0, 0)] + ring1 + list(best_combo)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        for inner_angle in [0.0, 30.0]:
            layouts.append((positions[:N], inner_angle))

    return layouts

def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    layouts = _build_layouts()

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    for positions, inner_angle in layouts:
        if time.time() > _DEADLINE - 10.0:
            break

        inner_hexes = [[pos[0], pos[1], inner_angle] for pos in positions]
        verts = _compute_all_vertices(inner_hexes)

        cx, cy, side, angle = _search_outer_angle(verts)

        if side < best_side:
            best_side = side
            best_inner = inner_hexes
            best_center = (cx, cy)
            best_outer_angle = angle

    # Polish the best layout
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        verts = _compute_all_vertices(best_inner)
        polished = _polish_with_nelder_mead(verts, best_center[0], best_center[1], best_outer_angle)
        if polished is not None:
            cx_p, cy_p, side_p, ang_p = polished
            if side_p < best_side:
                best_side = side_p
                best_center = (cx_p, cy_p)
                best_outer_angle = ang_p

        # Second polish pass
        if time.time() < _DEADLINE - 10.0:
            polished2 = _polish_with_nelder_mead(verts, best_center[0], best_center[1], best_outer_angle)
            if polished2 is not None:
                cx_p2, cy_p2, side_p2, ang_p2 = polished2
                if side_p2 < best_side:
                    best_side = side_p2
                    best_center = (cx_p2, cy_p2)
                    best_outer_angle = ang_p2

        # Third polish pass if time permits
        if time.time() < _DEADLINE - 5.0:
            polished3 = _polish_with_nelder_mead(verts, best_center[0], best_center[1], best_outer_angle)
            if polished3 is not None:
                cx_p3, cy_p3, side_p3, ang_p3 = polished3
                if side_p3 < best_side:
                    best_side = side_p3
                    best_center = (cx_p3, cy_p3)
                    best_outer_angle = ang_p3

    # Fallback
    if best_inner is None:
        scale = 1.000001
        u = (_SQRT3 * scale, 0.0)
        v = (_SQRT3 / 2 * scale, 1.5 * scale)
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        best_inner = [[coords_sorted[i][0], coords_sorted[i][1], 0.0] for i in range(N)]
        verts = _compute_all_vertices(best_inner)
        cx, cy, side, angle = _search_outer_angle(verts)
        best_side = side
        best_center = (cx, cy)
        best_outer_angle = angle

    return best_inner, best_center, best_side, best_outer_angle
