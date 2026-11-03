import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 145.0
SQRT3 = math.sqrt(3.0)


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side(theta, centers, theta_outer):
    """Minimal outer side length containing all inner hexagon vertices."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem * 2.0 / SQRT3


def _pairwise_penalty(centers):
    """Penalty for pairs of centers closer than 2 (overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    """Objective: outer side length + large penalty for overlap."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer, tol=1e-6):
    """Check non-overlap and containment."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    s = _outer_side(theta, centers, theta_outer)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    Use LP to find the optimal outer hexagon center and inradius
    for given inner hexagons and outer angle.
    Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    n_pts = len(all_vertices)
    c = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)

        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None


def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Find best outer hexagon angle and center via LP.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse search over 0..30 degrees (hexagonal symmetry)
    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Fine search around best
    if best_angle >= 0 and time.time() < start + time_limit:
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    # Ultra-fine search
    if best_angle >= 0 and time.time() < start + time_limit:
        fine_lo = max(0.0, best_angle - 0.05)
        fine_hi = min(30.0, best_angle + 0.05)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.001)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle


def _make_init(seed):
    """Generate initial parameters for optimization."""
    rng = np.random.RandomState(seed)
    mode = seed % 8
    sqrt3 = math.sqrt(3.0)

    if mode == 0:
        # Standard hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        # Offset hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        # Compact cluster on triangular lattice
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float)
    elif mode == 3:
        # Triangular lattice, 11 closest points
        u = (sqrt3, 0.0)
        v = (sqrt3 / 2, 1.5)
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                candidates.append((x * x + y * y, x, y))
        candidates.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in candidates[:N]])
    elif mode == 4:
        # Random with jitter around lattice
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float) + rng.uniform(-0.1, 0.1, size=(N, 2))
    elif mode == 5:
        # Slightly perturbed hexagonal arrangement
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + rng.uniform(-0.1, 0.1)
            r = 2.0 + rng.uniform(-0.1, 0.1)
            centers[i + 1] = [r * math.cos(ang), r * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0 + rng.uniform(-0.1, 0.1)
            r = 3.5 + rng.uniform(-0.15, 0.15)
            centers[i + 7] = [r * math.cos(ang), r * math.sin(ang)]
    elif mode == 6:
        # Hexagonal lattice with slight radial compression
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.3 * math.cos(ang), 3.3 * math.sin(ang)]
    else:
        # Hexagonal lattice with angular shift
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        shift = rng.uniform(-0.1, 0.1)
        for i in range(6):
            ang = i * math.pi / 3.0 + shift
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0 + shift
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter=400):
    """Single L-BFGS-B optimization run."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res
    except Exception:
        return None


def _polish(params, theta_outer, time_budget=10.0):
    """High-precision polish with multiple methods and tight tolerances."""
    start = time.time()
    cur_best = _objective(params, theta_outer)
    best_params = params.copy()

    # Step 1: L-BFGS-B with tight tolerances
    if time.time() - start < time_budget * 0.3:
        try:
            res = minimize(_objective, params, args=(theta_outer,),
                           method='L-BFGS-B',
                           options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < cur_best:
                cur_best = res.fun
                best_params = res.x
        except Exception:
            pass

    # Step 2: SLSQP with exact objective
    if time.time() - start < time_budget * 0.6:
        try:
            res = minimize(_objective, best_params, args=(theta_outer,),
                           method='SLSQP', options={'maxiter': 600, 'ftol': 1e-16, 'maxls': 100})
            if res.fun < cur_best:
                cur_best = res.fun
                best_params = res.x
        except Exception:
            pass

    # Step 3: Another L-BFGS-B pass
    if time.time() - start < time_budget * 0.85:
        try:
            res = minimize(_objective, best_params, args=(theta_outer,),
                           method='L-BFGS-B',
                           options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < cur_best:
                cur_best = res.fun
                best_params = res.x
        except Exception:
            pass

    # Step 4: Final SLSQP
    if time.time() - start < time_budget:
        try:
            res = minimize(_objective, best_params, args=(theta_outer,),
                           method='SLSQP', options={'maxiter': 400, 'ftol': 1e-17})
            if res.fun < cur_best:
                cur_best = res.fun
                best_params = res.x
        except Exception:
            pass

    return best_params


def _lp_refine(centers, theta, time_limit=6.0):
    """
    Use LP to find the best outer hexagon (center, angle) for given inner hexagons.
    Returns (cx, cy, side, angle_deg) or None.
    """
    inner_hexes = [[float(centers[i, 0]), float(centers[i, 1]),
                    float(math.degrees(theta))] for i in range(N)]
    return _find_best_outer_lp(inner_hexes, time_limit=time_limit)


def construct():
    best_score = float('inf')
    best_result = None  # (centers, theta, theta_o, cx, cy)

    # Phase 1: Multi-start continuous optimization with fixed outer angle
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0]
    seed = 0
    maxiter = 350

    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 40:  # reserve 40s for LP refinement + polish
            init_params = _make_init(seed)
            seed += 1
            res = _optimize_one(init_params, theta_o, maxiter)
            if res is None:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers.copy(), theta, theta_o, 0.0, 0.0)

    # Phase 2: LP-based outer hexagon optimization for the best inner configuration
    if best_result is not None and time.time() < DEADLINE - 20:
        centers, theta, _, _, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=12.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            # Verify validity with LP-found outer hexagon
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad, cx, cy)

    # Phase 3: Polish the best solution (multiple outer rotations for polish)
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, theta, theta_o, cx, cy = best_result
        for theta_o_try in [theta_o, 0.0, math.pi / 6.0, math.pi / 3.0]:
            if time.time() > DEADLINE - 10:
                break
            params = np.concatenate([centers.reshape(-1), [theta]])
            polished = _polish(params, theta_o_try, time_budget=8.0)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            if _is_valid(centers_p, theta_p, theta_o_try):
                s_p = _outer_side(theta_p, centers_p, theta_o_try)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o_try, 0.0, 0.0)

    # Phase 4: Final LP refinement on polished solution
    if best_result is not None and time.time() < DEADLINE - 5:
        centers, theta, _, _, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=8.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad, cx, cy)

        # Phase 5: One more polish after LP
        if time.time() < DEADLINE - 2:
            centers, theta, theta_o, cx, cy = best_result
            params = np.concatenate([centers.reshape(-1), [theta]])
            polished = _polish(params, theta_o, time_budget=5.0)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            if _is_valid(centers_p, theta_p, theta_o):
                s_p = _outer_side(theta_p, centers_p, theta_o)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o, 0.0, 0.0)

    # Phase 6: Final LP after last polish
    if best_result is not None and time.time() < DEADLINE - 0.5:
        centers, theta, _, _, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=4.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad, cx, cy)

    if best_result is None:
        # Fallback: return something (may be invalid)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, theta_o, cx, cy = best_result
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(s), float(math.degrees(theta_o))
