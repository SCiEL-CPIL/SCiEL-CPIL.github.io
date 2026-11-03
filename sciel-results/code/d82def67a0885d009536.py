import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """Exact LP: min R s.t. hexagon centered (cx,cy) of 'inradius' R encloses verts.
    Returns (cx, cy, side) where side = R * 2 / sqrt(3)."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    vdots = verts @ normals.T  # (M, 6)

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
                               'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14})
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _smooth_side_for_center(verts, cx, cy, outer_angle_deg, beta):
    """Smooth (log-sum-exp) estimate of the minimal inradius R for a fixed center."""
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    vdots = verts @ normals.T
    dots_center = normals @ np.array([cx, cy])

    max_vals = np.max(vdots, axis=0) - dots_center
    min_vals = np.min(vdots, axis=0) - dots_center

    # Required inradius in direction j is max(max_vals[j], -min_vals[j])
    R_j = np.maximum(max_vals, -min_vals)

    m = np.max(R_j)
    if m > 1e12:
        return 1e12
    s = np.sum(np.exp((R_j - m) / beta))
    R_smooth = m + beta * math.log(s)
    return R_smooth * _INVSQRT3 * 2.0


def _smooth_side_optimized(verts, outer_angle_deg, beta, cx0=0.0, cy0=0.0):
    """Optimize center for a fixed outer angle using the smooth objective."""
    def obj(p):
        return _smooth_side_for_center(verts, p[0], p[1], outer_angle_deg, beta)
    try:
        res = minimize(obj, np.array([cx0, cy0]), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            return res.fun, res.x[0], res.x[1]
    except Exception:
        pass
    return None, cx0, cy0


def _objective_for_angle(verts, outer_angle_deg, beta):
    """Smooth side length after optimizing center for this outer angle."""
    val, _, _ = _smooth_side_optimized(verts, outer_angle_deg, beta)
    if val is None:
        return 1e12
    return val


def _build_positions(layout_idx):
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    def lattice(ti, tj):
        return (ti * u[0] + tj * v[0], ti * u[1] + tj * v[1])

    if layout_idx == 0:
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        return [lattice(a, b) for a, b in target[:N]]

    elif layout_idx == 1:
        # 3-5-3
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y1))
        y2 = 0.0
        for k in range(-2, 3):
            positions.append((k * sqrt3 * scale, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y3))
        return positions[:N]

    elif layout_idx == 2:
        # 4-3-4 shifted
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            positions.append((k * sqrt3 * scale, y1))
        y2 = 0.0
        for k in range(-1, 2):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            positions.append((k * sqrt3 * scale, y3))
        return positions[:N]

    elif layout_idx == 3:
        # 3-4-4 rows
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y1))
        y2 = 0.0
        for k in range(-1, 3):
            positions.append((k * sqrt3 * scale, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y3))
        return positions[:N]

    elif layout_idx == 4:
        # 2-4-2-3
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            positions.append((k * sqrt3 * scale - sqrt3 / 2 * scale, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            positions.append((k * sqrt3 * scale, y2))
        y3 = 0.0
        for k in range(-1, 1):
            positions.append((k * sqrt3 * scale + sqrt3 / 2 * scale, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            positions.append((k * sqrt3 * scale, y4))
        return positions[:N]

    elif layout_idx == 5:
        # compact symmetric ring2 pick
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        pick = [ring2[0], ring2[2], ring2[5], ring2[7]]
        target.extend(pick)
        return [lattice(a, b) for a, b in target[:N]]

    elif layout_idx == 6:
        # compact alternate ring2 pick
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
                 (-1, -1), (0, -2), (1, -2), (2, -1)]
        pick = [ring2[1], ring2[3], ring2[6], ring2[8]]
        target.extend(pick)
        return [lattice(a, b) for a, b in target[:N]]

    else:
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                coords.append(lattice(i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return coords_sorted[:N]


def _relax_positions(positions, inner_angle, iters=30, step=0.02):
    """Light local relaxation: move inner hexagons slightly to shrink the
    bounding hexagon (smooth objective), keeping them disjoint (unit spacing)."""
    pts = np.array(positions, dtype=float)  # (N,2)
    # Precompute pair distances for overlap check
    def smooth_size(p):
        verts = np.vstack([_hex_vertices(p[i, 0], p[i, 1], inner_angle) for i in range(N)])
        return _smooth_side_for_center(verts, 0.0, 0.0, 0.0, 8.0)

    cur = smooth_size(pts)
    for it in range(iters):
        improved = False
        for i in range(N):
            for dx in (-step, 0.0, step):
                for dy in (-step, 0.0, step):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    cand = pts.copy()
                    cand[i, 0] += dx
                    cand[i, 1] += dy
                    # check disjointness: min distance between centers >= sqrt(3)
                    ok = True
                    for j in range(N):
                        if j == i:
                            continue
                        d = math.hypot(cand[i, 0] - cand[j, 0], cand[i, 1] - cand[j, 1])
                        if d < _SQRT3 * 0.999:
                            ok = False
                            break
                    if not ok:
                        continue
                    val = smooth_size(cand)
                    if val < cur - 1e-9:
                        cur = val
                        pts = cand
                        improved = True
        if not improved:
            break
    return pts, cur


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0

    num_layouts = 8
    inner_angles = [0.0, 30.0, 15.0, -15.0]

    for layout_idx in range(num_layouts):
        if time.time() > _DEADLINE - 12.0:
            break

        positions = _build_positions(layout_idx)
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 12.0:
                break

            # Optionally relax positions (only for the most compact layouts)
            use_relax = layout_idx in (0, 5, 6)
            if use_relax and time.time() < _DEADLINE - 30.0:
                pts, _ = _relax_positions(positions, inner_angle, iters=15, step=0.02)
                positions = [tuple(p) for p in pts]

            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle)
                                      for px, py in positions])

            # ---- Coarse grid over outer angle with exact LP ----
            coarse = []
            for oa in np.arange(0.0, 30.01, 0.5):
                if time.time() > _DEADLINE - 12.0:
                    break
                r = _min_enclosing_hex_side(all_vertices, oa)
                if r:
                    coarse.append((r[2], oa, r[0], r[1]))

            if not coarse:
                continue
            coarse.sort(key=lambda x: x[0])
            best_local_side = coarse[0][0]
            best_local_angle = coarse[0][1]
            best_local_center = (coarse[0][2], coarse[0][3])

            # ---- Smooth refinement of outer angle (Nelder-Mead) ----
            if time.time() < _DEADLINE - 20.0:
                # Use a moderate beta for the smooth objective
                beta = 12.0
                # Start from the best coarse angle
                x0 = np.array([best_local_angle])

                def obj(p):
                    return _objective_for_angle(all_vertices, p[0], beta)

                try:
                    res = minimize(obj, x0, method='Nelder-Mead',
                                   options={'maxiter': 60, 'xatol': 1e-4,
                                            'fatol': 1e-9,
                                            'initial_simplex': np.array([[best_local_angle],
                                                                          [best_local_angle + 0.5]])})
                    if res.x[0] is not None:
                        cand_angle = float(res.x[0]) % 30.0
                        r = _min_enclosing_hex_side(all_vertices, cand_angle)
                        if r and r[2] < best_local_side:
                            best_local_side = r[2]
                            best_local_angle = cand_angle
                            best_local_center = (r[0], r[1])
                except Exception:
                    pass

                # Golden-section polish around the best angle
                lo = max(0.0, best_local_angle - 1.0)
                hi = min(30.0, best_local_angle + 1.0)
                if hi - lo > 1e-4 and time.time() < _DEADLINE - 25.0:
                    try:
                        rgs = minimize_scalar(lambda a: _objective_for_angle(all_vertices, a, beta),
                                              bounds=(lo, hi), method='bounded',
                                              options={'xatol': 1e-4, 'maxiter': 40})
                        cand_angle = float(rgs.x)
                        r = _min_enclosing_hex_side(all_vertices, cand_angle)
                        if r and r[2] < best_local_side:
                            best_local_side = r[2]
                            best_local_angle = cand_angle
                            best_local_center = (r[0], r[1])
                    except Exception:
                        pass

            # ---- Fine exact LP grid around the best angle ----
            if best_local_side < best_side and time.time() < _DEADLINE - 25.0:
                lo = max(0.0, best_local_angle - 0.3)
                hi = min(30.0, best_local_angle + 0.3)
                for oa in np.arange(lo, hi + 1e-9, 0.02):
                    if time.time() > _DEADLINE - 20.0:
                        break
                    r = _min_enclosing_hex_side(all_vertices, oa)
                    if r and r[2] < best_local_side:
                        best_local_side = r[2]
                        best_local_angle = oa
                        best_local_center = (r[0], r[1])

            # ---- Ultra-fine exact LP grid ----
            if best_local_side < best_side and time.time() < _DEADLINE - 30.0:
                lo = max(0.0, best_local_angle - 0.05)
                hi = min(30.0, best_local_angle + 0.05)
                for oa in np.arange(lo, hi + 1e-9, 0.002):
                    if time.time() > _DEADLINE - 25.0:
                        break
                    r = _min_enclosing_hex_side(all_vertices, oa)
                    if r and r[2] < best_local_side:
                        best_local_side = r[2]
                        best_local_angle = oa
                        best_local_center = (r[0], r[1])

            # ---- Commit if improved ----
            if best_local_side < best_side:
                r_final = _min_enclosing_hex_side(all_vertices, best_local_angle)
                if r_final:
                    cx, cy, side = r_final
                    if side < best_side:
                        best_side = side
                        best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                        best_center = (cx, cy)
                        best_outer_angle = best_local_angle
                        best_inner_angle = inner_angle

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Safety margin for floating-point / evaluator strictness
    safe_side = best_side * 1.0005

    return best_inner, best_center, safe_side, best_outer_angle
