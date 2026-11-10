# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START = time.time()
_DEADLINE = 600.0
_SAFETY = 45.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(np.arange(6) * 60.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _outer_vertices(ox, oy, side, a_deg):
    return _hex_vertices(ox, oy, side, a_deg)


def _min_gap_pair(v1, v2):
    """Minimum separation between two convex hexagons (arrays of 6 (x,y) vertices).
    Returns the minimum over the 12 edge normals of the gap (positive = separated).
    If polygons overlap in projection on every normal, returns a negative value."""
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if v1.ndim == 1:
        v1 = v1.reshape(-1, 2)
    if v2.ndim == 1:
        v2 = v2.reshape(-1, 2)

    def normals(v):
        e = np.roll(v, -1, axis=0) - v
        n = np.column_stack([e[:, 1], -e[:, 0]])
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
        return n

    n1 = normals(v1)
    n2 = normals(v2)
    best = -np.inf
    for n in np.vstack([n1, n2]):
        p1 = v1 @ n
        p2 = v2 @ n
        gap = min(p2.min() - p1.max(), p1.min() - p2.max())
        if gap > best:
            best = gap
    return best


def _inside_hexagon(pts, ov):
    """Check all pts (N,2) are inside convex hexagon ov (6,2) CCW.
    Returns True if all inside (with tolerance)."""
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (pts[:, 1] - p1[1]) - (p2[1] - p1[1]) * (pts[:, 0] - p1[0])
        if np.any(cross < -1e-6):
            return False
    return True


def _validate(inner, outer, side):
    # inner: (11,3) (x,y,angle_deg); outer: (3,) (x,y,angle_deg)
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    # vertices of each inner hex
    vs = []
    for (x, y, ang) in inner:
        vs.append(_hex_vertices(x, y, 1.0, ang))
    # pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            g = _min_gap_pair(vs[i], vs[j])
            if g <= 1e-6:
                return False
    # containment: shift by outer center
    ox, oy, oang = outer
    ov = _outer_vertices(ox, oy, side, oang)
    # all inner vertices must be inside
    for v in vs:
        if not _inside_hexagon(v, ov):
            return False
    # edge lengths within 1e-8 of 1
    for v in vs:
        e = np.roll(v, -1, axis=0) - v
        L = np.linalg.norm(e, axis=1)
        if np.any(np.abs(L - 1.0) > 1e-8):
            return False
    return True


def _required_side_for_center(positions, N, apothem_factor, center):
    # positions: (M,2) all inner vertices
    d = positions - center
    proj = d @ N.T  # (M,6)
    s = proj.max() / apothem_factor
    return s


def _all_inner_vertices(positions, angles):
    """Compute all 66 vertices of 11 unit hexagons given centers and angles."""
    allv = []
    for i in range(11):
        allv.append(_hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i]))
    return np.vstack(allv)


def _objective_positions(x, oang_deg, penalty_weight=1e5):
    """Objective for optimizing inner positions. x is (22,) = 11*(x,y)."""
    p = x.reshape(11, 2)
    angles = np.zeros(11)
    allv = _all_inner_vertices(p, angles)

    # Compute required side with center at centroid
    a = np.deg2rad(oang_deg)
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    apothem_factor = np.sqrt(3.0) / 2.0
    c = np.mean(p, axis=0)
    s = _required_side_for_center(allv, N, apothem_factor, c)

    # Penalty for overlaps
    pen = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            g = _min_gap_pair(allv[i * 6:(i + 1) * 6], allv[j * 6:(j + 1) * 6])
            if g < 1e-6:
                pen += (1e-6 - g) ** 2

    return s + penalty_weight * pen


def _solve_enclosing_center(allv, oang_deg):
    """Given all inner vertices, find optimal outer center to minimize side."""
    a = np.deg2rad(oang_deg)
    nangles = a + np.deg2rad(30.0 + np.arange(6) * 60.0)
    N = np.column_stack([np.cos(nangles), np.sin(nangles)])
    apothem_factor = np.sqrt(3.0) / 2.0

    def obj(c):
        return _required_side_for_center(allv, N, apothem_factor, c)

    c0 = np.mean(allv, axis=0)
    res = minimize(obj, c0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
    best_c = res.x
    best_s = _required_side_for_center(allv, N, apothem_factor, best_c)
    return best_c, best_s


def _build_inner(positions, angles):
    positions = np.asarray(positions)
    if positions.ndim == 1:
        positions = positions.reshape(11, 2)
    angles = np.asarray(angles)
    if angles.ndim == 0:
        angles = np.full(11, float(angles))
    inner = np.zeros((11, 3))
    for i in range(11):
        inner[i, 0] = positions[i, 0]
        inner[i, 1] = positions[i, 1]
        inner[i, 2] = angles[i]
    return inner


def _initial_positions():
    """Initial hexagonal close packing: 1 center + 6 ring1 + 4 from ring2."""
    d = np.sqrt(3.0)
    pos = np.zeros((11, 2))
    pos[0] = [0.0, 0.0]
    for k in range(6):
        ang = np.deg2rad(k * 60.0)
        pos[k + 1] = [d * np.cos(ang), d * np.sin(ang)]
    # 4 extra at ring 2, alternating
    extra_angles = [0, 120, 240, 300]
    for k, a_deg in enumerate(extra_angles):
        ang = np.deg2rad(a_deg)
        pos[7 + k] = [2 * d * np.cos(ang), 2 * d * np.sin(ang)]
    angles = np.zeros(11)
    return pos, angles


def _initial_positions_alternate():
    """Alternative initial positions: more compact arrangement."""
    d = np.sqrt(3.0)
    pos = np.zeros((11, 2))
    pos[0] = [0.0, 0.0]
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(k * 60.0)
        pos[k + 1] = [d * np.cos(ang), d * np.sin(ang)]
    # 4 extra: place in gaps of ring 2
    # Ring 2 positions are at distance 2d at angles 0,60,120,180,240,300
    # The "gap" positions between ring 1 and ring 2 are at distance sqrt(3)*d = 3
    # at angles 30, 90, 150, 210, 270, 330
    gap_d = 3.0
    gap_angles = [30, 90, 150, 210, 270, 330]
    # Pick 4 of the 6 gap positions
    for k in range(4):
        ang = np.deg2rad(gap_angles[k])
        pos[7 + k] = [gap_d * np.cos(ang), gap_d * np.sin(ang)]
    angles = np.zeros(11)
    return pos, angles


def hexagon_packing_11():
    best = None
    best_side = np.inf
    start_time = time.time()

    def time_left():
        return _DEADLINE - _SAFETY - (time.time() - start_time)

    # Try multiple initial configurations and outer orientations
    initial_configs = [
        _initial_positions,
        _initial_positions_alternate,
    ]
    outer_angles = [0.0, 30.0, 15.0, 45.0, 60.0, 10.0, 20.0, 50.0]

    for init_fn in initial_configs:
        if time_left() < 30:
            break
        pos, angles = init_fn()
        for oang in outer_angles:
            if time_left() < 30:
                break

            # First: optimize enclosing center for fixed positions
            allv = _all_inner_vertices(pos, angles)
            c, s = _solve_enclosing_center(allv, oang)
            inner = _build_inner(pos, angles)
            outer = np.array([c[0], c[1], oang])
            if _validate(inner, outer, s):
                if s < best_side:
                    best_side = s
                    best = (inner.copy(), outer.copy(), float(s))

            # Then: optimize positions to reduce side
            if time_left() < 60:
                continue

            def obj(x):
                return _objective_positions(x, oang)

            x0 = pos.flatten()
            try:
                res = minimize(obj, x0, method='Nelder-Mead',
                               options={'maxiter': 15000, 'xatol': 1e-8, 'fatol': 1e-10})
                p_opt = res.x.reshape(11, 2)
            except Exception:
                p_opt = pos

            # Recompute with optimized positions
            allv_opt = _all_inner_vertices(p_opt, angles)
            c_opt, s_opt = _solve_enclosing_center(allv_opt, oang)
            inner_opt = _build_inner(p_opt, angles)
            outer_opt = np.array([c_opt[0], c_opt[1], oang])
            if _validate(inner_opt, outer_opt, s_opt):
                if s_opt < best_side:
                    best_side = s_opt
                    best = (inner_opt.copy(), outer_opt.copy(), float(s_opt))

    # If we have time, try a more global optimization
    if best is not None and time_left() > 120:
        # Use differential evolution for a broader search
        pos_best = best[0][:, :2]
        angles_best = best[0][:, 2]
        oang_best = best[1][2]

        # Define bounds: positions in a reasonable range
        bounds = [(-5, 5)] * 22

        def de_obj(x):
            return _objective_positions(x, oang_best, penalty_weight=1e4)

        try:
            de_res = differential_evolution(
                de_obj, bounds, seed=42, maxiter=200, tol=1e-8,
                popsize=15, mutation=(0.3, 1.0), recombination=0.7,
                polish=True, init='latinhypercube'
            )
            p_de = de_res.x.reshape(11, 2)
            allv_de = _all_inner_vertices(p_de, angles_best)
            c_de, s_de = _solve_enclosing_center(allv_de, oang_best)
            inner_de = _build_inner(p_de, angles_best)
            outer_de = np.array([c_de[0], c_de[1], oang_best])
            if _validate(inner_de, outer_de, s_de):
                if s_de < best_side:
                    best_side = s_de
                    best = (inner_de.copy(), outer_de.copy(), float(s_de))
        except Exception:
            pass

    # Final local refinement of the best solution
    if best is not None and time_left() > 60:
        pos_final = best[0][:, :2]
        angles_final = best[0][:, 2]
        oang_final = best[1][2]

        def final_obj(x):
            return _objective_positions(x, oang_final, penalty_weight=1e6)

        try:
            res_final = minimize(final_obj, pos_final.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12})
            p_final = res_final.x.reshape(11, 2)
            allv_final = _all_inner_vertices(p_final, angles_final)
            c_final, s_final = _solve_enclosing_center(allv_final, oang_final)
            inner_final = _build_inner(p_final, angles_final)
            outer_final = np.array([c_final[0], c_final[1], oang_final])
            if _validate(inner_final, outer_final, s_final):
                if s_final < best_side:
                    best_side = s_final
                    best = (inner_final.copy(), outer_final.copy(), float(s_final))
        except Exception:
            pass

    if best is None:
        # Fallback to a known-valid loose packing
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ])
        outer = np.array([0, 0, 0])
        return inner, outer, 8.0

    return best


# EVOLVE-BLOCK-END
