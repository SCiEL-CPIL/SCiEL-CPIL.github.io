import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.spatial import ConvexHull

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


def _hex_area(cx, cy, angle_deg):
    """Area of a unit regular hexagon (constant)."""
    return (3.0 * _SQRT3 / 2.0)


def _point_in_hex(px, py, cx, cy, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle."""
    dx = px - cx
    dy = py - cy
    a0 = math.radians(angle_deg)
    for j in range(6):
        ang = a0 + j * math.pi / 3.0
        nx = math.cos(ang + math.pi / 2)
        ny = math.sin(ang + math.pi / 2)
        if dx * nx + dy * ny > 1.0:
            return False
    return True


def _hexagons_overlap(v1, v2, tol=1e-9):
    """Check if two hexagons (each with 6 vertices) overlap using SAT."""
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            edge = verts[j] - verts[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
    return True


def _all_disjoint(hex_list, tol=1e-9):
    """Check if all hexagons in the list are pairwise disjoint."""
    n = len(hex_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(hex_list[i], hex_list[j], tol):
                return False
    return True


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


def _smooth_objective_grad(verts, outer_angle_deg, beta=100.0):
    """
    Smooth surrogate for the outer side length using log-sum-exp, with gradient.
    Returns (objective_value, gradient_scalar).
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

    weights = np.exp(beta * shifted)
    weights /= weights.sum()

    max_idx = np.argmax(vdots, axis=0)  # shape (6,)
    min_idx = np.argmin(vdots, axis=0)  # shape (6,)

    dn_dtheta = np.column_stack([-np.sin(angles), np.cos(angles)])

    v_max = verts[max_idx]  # shape (6, 2)
    v_min = verts[min_idx]  # shape (6, 2)

    dR_dtheta = 0.5 * (v_max @ dn_dtheta.T - v_min @ dn_dtheta.T)  # shape (6,)

    dLSE_dtheta = np.sum(weights * dR_dtheta)
    dside_dtheta = 2.0 * _INVSQRT3 * dLSE_dtheta

    dside_ddeg = dside_dtheta * math.pi / 180.0

    return side, dside_ddeg


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

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
        # Center + 6 ring + 4 more spread out
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        # Pick 4 from ring2 that are spread out
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        # Choose alternating ones for better spread
        chosen = [ring2[0], ring2[2], ring2[5], ring2[7]]
        target.extend(chosen)
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # Linear-ish: 2 rows of 5 + 1
        positions = []
        y1 = -0.75 * scale
        for k in range(5):
            x = (k - 2) * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.75 * scale
        for k in range(5):
            x = (k - 2) * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        positions.append((0.0, 2.25 * scale))
        return positions[:N]
    elif layout_idx == 7:
        # Cross pattern
        positions = []
        # Center
        positions.append((0.0, 0.0))
        # 4 arms
        for ang_deg in [0, 90, 180, 270]:
            a = math.radians(ang_deg)
            positions.append((2.0 * scale * math.cos(a), 2.0 * scale * math.sin(a)))
        # 4 diagonals
        for ang_deg in [45, 135, 225, 315]:
            a = math.radians(ang_deg)
            positions.append((1.8 * scale * math.cos(a), 1.8 * scale * math.sin(a)))
        return positions[:N]
    else:
        # Fallback: nearest to origin
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _optimize_inner_positions(positions, angles, outer_angle_deg, time_limit):
    """
    Local optimization of inner hexagon positions to minimize enclosing hexagon side.
    Uses gradient-free optimization (Nelder-Mead) on the smooth surrogate.
    Returns optimized positions, angles, and best side.
    """
    def _get_all_vertices(pos_list, ang_list):
        return np.vstack(
            [_hex_vertices(pos_list[i][0], pos_list[i][1], ang_list[i]) 
             for i in range(len(pos_list))]
        )

    def _check_valid(pos_list, ang_list):
        hex_list = [_hex_vertices(pos_list[i][0], pos_list[i][1], ang_list[i]) 
                    for i in range(len(pos_list))]
        return _all_disjoint(hex_list, tol=1e-8)

    def _objective(params):
        if time.time() > time_limit:
            return 1e10
        pos_list = [(params[2*i], params[2*i+1]) for i in range(N)]
        ang_list = [angles[i] for i in range(N)]
        
        if not _check_valid(pos_list, ang_list):
            return 1e8
        
        verts = _get_all_vertices(pos_list, ang_list)
        center_offset = np.mean(verts, axis=0)
        centered = verts - center_offset
        side, _ = _smooth_objective_grad(centered, outer_angle_deg, beta=50.0)
        return side

    # Initial parameters
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    try:
        res = minimize(
            _objective,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
        )
        if res.fun < 1e7:
            pos_opt = [(res.x[2*i], res.x[2*i+1]) for i in range(N)]
            if _check_valid(pos_opt, angles):
                return pos_opt, angles, res.fun
    except Exception:
        pass
    
    return positions, angles, 1e10


def _optimize_outer_angle(verts, initial_angle, time_limit):
    """
    Optimize the outer hexagon angle using smooth surrogate + exact refinement.
    Returns (best_angle, best_side).
    """
    best_side = 1e10
    best_angle = initial_angle

    # Step 1: Smooth optimization via L-BFGS-B with analytic gradient
    try:
        def obj_and_grad(a):
            s, g = _smooth_objective_grad(verts, a[0], beta=100.0)
            return s, np.array([g])

        res_opt = minimize(
            obj_and_grad,
            x0=[initial_angle],
            method='L-BFGS-B',
            bounds=[(0.0, 30.0)],
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if res_opt.success:
            opt_angle = float(np.clip(res_opt.x[0], 0.0, 30.0))
            if time.time() < time_limit:
                exact_side = _min_enclosing_hex_side(verts, opt_angle)
                if exact_side is not None:
                    if exact_side[2] < best_side:
                        best_side = exact_side[2]
                        best_angle = opt_angle
    except Exception:
        pass

    # Step 2: Bounded scalar minimization with exact objective
    lo = max(0.0, best_angle - 5.0)
    hi = min(30.0, best_angle + 5.0)
    try:
        res_scalar = minimize_scalar(
            lambda a: _min_enclosing_hex_side(verts, a)[2] if _min_enclosing_hex_side(verts, a) else 1e6,
            bounds=(lo, hi),
            method='bounded',
            options={'xatol': 1e-12, 'maxiter': 300}
        )
        if res_scalar.success:
            if res_scalar.fun < best_side:
                best_side = res_scalar.fun
                best_angle = float(res_scalar.x)
    except Exception:
        pass

    # Step 3: Golden-section polish with exact objective
    if time.time() < time_limit:
        for delta_max, tol in [(0.01, 1e-10), (0.002, 1e-12)]:
            if time.time() > time_limit:
                break
            lo2 = max(0.0, best_angle - delta_max)
            hi2 = min(30.0, best_angle + delta_max)
            if hi2 <= lo2:
                continue
            gr = (math.sqrt(5) - 1) / 2
            a = lo2
            b = hi2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            fc = _min_enclosing_hex_side(verts, c)
            fd = _min_enclosing_hex_side(verts, d)
            fc = fc[2] if fc else 1e6
            fd = fd[2] if fd else 1e6
            while (b - a) > tol and time.time() < time_limit:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = _min_enclosing_hex_side(verts, c)
                    fc = fc[2] if fc else 1e6
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = _min_enclosing_hex_side(verts, d)
                    fd = fd[2] if fd else 1e6
            mid = (a + b) / 2
            res = _min_enclosing_hex_side(verts, mid)
            if res:
                s = res[2]
                if s < best_side:
                    best_side = s
                    best_angle = float(mid)

    return best_angle, best_side


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 8
    layouts = [_build_positions(i) for i in range(num_layouts)]

    inner_angles = [0.0, 30.0, 15.0, -15.0]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break

        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 12.0:
                break

            angles = [inner_angle] * N
            hex_list = [_hex_vertices(px, py, inner_angle) for px, py in positions]
            
            if not _all_disjoint(hex_list, tol=1e-8):
                continue

            all_vertices = np.vstack(hex_list)
            center_offset = np.mean(all_vertices, axis=0)
            centered_vertices = all_vertices - center_offset

            # Try optimizing inner positions
            time_left = _DEADLINE - time.time()
            if time_left > 30.0:
                opt_positions, opt_angles, opt_side = _optimize_inner_positions(
                    positions, angles, 0.0, _DEADLINE - 20.0
                )
                if opt_side < 1e7:
                    positions = opt_positions
                    angles = opt_angles
                    hex_list = [_hex_vertices(positions[i][0], positions[i][1], angles[i]) 
                                for i in range(N)]
                    if _all_disjoint(hex_list, tol=1e-8):
                        all_vertices = np.vstack(hex_list)
                        center_offset = np.mean(all_vertices, axis=0)
                        centered_vertices = all_vertices - center_offset

            # Coarse angle search
            coarse_best = 1e10
            coarse_best_angle = 0.0
            coarse_results = []
            for outer_angle_deg in np.arange(0, 30.25, 0.5):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(centered_vertices, outer_angle_deg)
                if res:
                    coarse_results.append((res[2], outer_angle_deg))
                    if res[2] < coarse_best:
                        coarse_best = res[2]
                        coarse_best_angle = outer_angle_deg

            if not coarse_results:
                continue

            coarse_results.sort(key=lambda x: x[0])
            candidate_starts = [coarse_results[0][1]]
            for s, a in coarse_results[1:5]:
                if abs(a - candidate_starts[0]) > 0.5:
                    candidate_starts.append(a)

            local_best_side = coarse_best
            local_best_angle = coarse_best_angle

            for start_a in candidate_starts:
                if time.time() > _DEADLINE - 15.0:
                    break
                ang, side = _optimize_outer_angle(centered_vertices, start_a, _DEADLINE - 18.0)
                if side < local_best_side:
                    local_best_side = side
                    local_best_angle = ang

            if local_best_side < best_side:
                res_final = _min_enclosing_hex_side(centered_vertices, local_best_angle)
                if res_final:
                    cx, cy, side = res_final
                    final_cx = cx + center_offset[0]
                    final_cy = cy + center_offset[1]
                    if side < best_side:
                        best_side = side
                        best_inner = [[positions[i][0], positions[i][1], angles[i]] 
                                      for i in range(N)]
                        best_center = (final_cx, final_cy)
                        best_outer_angle = local_best_angle

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
