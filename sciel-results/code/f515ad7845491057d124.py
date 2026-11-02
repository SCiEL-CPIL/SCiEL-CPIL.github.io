"""Heilbronn convex regions n=14 – centrally-symmetric DE + NM polish."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
M = N // 2  # 7 independent points for central symmetry

# ─── fast area helpers ────────────────────────────────────────────────────────

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area (not doubled) for N points."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

# ─── central-symmetric parameterization ───────────────────────────────────────

def _decode_central_symmetric(params):
    """
    params: shape (M, 2) – M independent points.
    Returns: shape (N, 2) – full centrally-symmetric set, sorted by angle from origin.
    Central symmetry: if p is in the set, -p is also in the set.
    """
    pts = np.vstack([params, -params])  # (N, 2)
    # Sort by angle from origin for consistent convex ordering
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    return pts[order]

def _objective_symmetric(params_flat):
    """Minimize hull_area / min_triangle_area for centrally-symmetric config."""
    params = params_flat.reshape(M, 2)
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return 1e10
    points = _decode_central_symmetric(params)
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_symmetric_soft(params_flat, beta=150.0):
    """Soft-min version for smoother gradient-based optimization."""
    params = params_flat.reshape(M, 2)
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return 1e10
    points = _decode_central_symmetric(params)
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

# ─── Cartesian objectives (for polish in full 28D space) ─────────────────────

def _objective_hard(points_flat):
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft_cart(points_flat, beta=150.0):
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

# ─── normalization ────────────────────────────────────────────────────────────

def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points

# ─── coordinate descent on Cartesian points ──────────────────────────────────

def _coordinate_descent(pts, deadline, max_rounds=30):
    pts = pts.copy()
    n = len(pts)
    step0 = 0.03
    for rnd in range(max_rounds):
        if time.time() > deadline - 6:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 6:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                best_val = _score(pts)
                best_pos = orig
                for _ in range(25):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig

                    if s_plus > best_val:
                        best_val = s_plus
                        best_pos = orig + step
                    elif s_minus > best_val:
                        best_val = s_minus
                        best_pos = orig - step

                    if s_plus > s_minus and s_plus > best_val - 1e-15:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus and s_minus > best_val - 1e-15:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-10:
                        break
                if best_pos != orig:
                    pts[i, coord] = best_pos
        if not improved:
            break
    return pts

# ─── initializations in symmetric space ───────────────────────────────────────

def _generate_sym_inits():
    """Generate diverse initializations in the M×2 symmetric parameter space."""
    inits = []
    rng = np.random.default_rng(42)

    # Regular M-gon (first half of regular 14-gon)
    angles_m = np.linspace(0, np.pi, M, endpoint=False)
    reg = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    inits.append(reg.flatten())

    # Rotated regular M-gons
    for rot in [0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6]:
        a = angles_m + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts.flatten())

    # Jittered regular M-gon
    for jitter in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        jittered = reg + rng.normal(0, jitter, size=(M, 2))
        inits.append(jittered.flatten())

    # Two-ring: inner + outer (symmetric pairs)
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for n_in in [3, 4, 5]:
            n_out = M - n_in
            a_in = np.linspace(0, np.pi, n_in, endpoint=False)
            a_out = np.linspace(0, np.pi, n_out, endpoint=False) + np.pi / max(n_out, 1)
            pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
            pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
            pts = np.vstack([pts_in, pts_out])
            inits.append(pts.flatten())

    # Elliptical
    for ax, ay in [(1.0, 0.5), (1.0, 0.7), (1.2, 0.8), (1.3, 0.9), (1.5, 0.7)]:
        a_e = np.linspace(0, np.pi, M, endpoint=False)
        pts = np.column_stack([ax * np.cos(a_e), ay * np.sin(a_e)])
        inits.append(pts.flatten())

    # Alternating radii
    for alt in [0.1, 0.2, 0.3, 0.4]:
        a_alt = np.linspace(0, np.pi, M, endpoint=False)
        radii = np.ones(M)
        radii[::2] = 1.0 + alt
        radii[1::2] = 1.0 - alt
        pts = np.column_stack([radii * np.cos(a_alt), radii * np.sin(a_alt)])
        inits.append(pts.flatten())

    # Random convex-like (sort by angle, keep in upper half for symmetry)
    for seed in range(10):
        rng_s = np.random.default_rng(seed * 1000 + 7)
        pts_raw = rng_s.uniform(-1, 1, size=(M, 2))
        # Ensure points are in a half-plane for convexity after symmetry
        # Project to upper half-plane
        for i in range(M):
            if pts_raw[i, 1] < 0:
                pts_raw[i, 1] = -pts_raw[i, 1]
        angles_c = np.arctan2(pts_raw[:, 1], pts_raw[:, 0])
        sorted_idx = np.argsort(angles_c)
        inits.append(pts_raw[sorted_idx].flatten())

    # Fibonacci-like
    golden_angle = 2.399963229728653
    for r_scale in [1.0, 0.8, 1.2]:
        pts_fib = np.zeros((M, 2))
        for i in range(M):
            angle = i * golden_angle % np.pi
            r = r_scale * np.sqrt((i + 0.5) / M)
            pts_fib[i, 0] = r * np.cos(angle)
            pts_fib[i, 1] = r * np.sin(angle)
        inits.append(pts_fib.flatten())

    # Pure random
    for seed in range(12):
        rng_s = np.random.default_rng(seed * 777 + 13)
        pts_rand = rng_s.uniform(-1, 1, size=(M, 2))
        angles_r = np.arctan2(pts_rand[:, 1], pts_rand[:, 0])
        sorted_idx = np.argsort(angles_r)
        inits.append(pts_rand[sorted_idx].flatten())

    return inits

# ─── main construct ───────────────────────────────────────────────────────────

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    best_points = None
    best_score = 0.0

    def _update(points):
        nonlocal best_points, best_score
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # ── Phase 1: Differential Evolution in symmetric (14D) space ──
    inits = _generate_sym_inits()
    # Build DE bounds: each coordinate in [-2, 2]
    bounds = [(-2.0, 2.0)] * (M * 2)

    remaining = deadline - time.time()
    if remaining > 20:
        try:
            de_result = differential_evolution(
                _objective_symmetric,
                bounds=bounds,
                seed=42,
                maxiter=200,
                popsize=15,
                tol=1e-10,
                mutation=(0.3, 0.9),
                recombination=0.8,
                polish=False,
                init='sobol',
                update_interval=1,
            )
            if de_result.success or de_result.fun < 1e9:
                pts_de = _decode_central_symmetric(de_result.x.reshape(M, 2))
                _update(pts_de)
        except Exception:
            pass

    # ── Phase 2: NM polish in symmetric space from best + diverse starts ──
    # Get current best in symmetric form
    if best_points is not None:
        # Try to recover symmetric params: take first M points sorted by angle
        bp = best_points.copy()
        angles_bp = np.arctan2(bp[:, 1], bp[:, 0])
        order_bp = np.argsort(angles_bp)
        bp_sorted = bp[order_bp]
        # First M of sorted (upper half roughly)
        sym_params_best = bp_sorted[:M]
    else:
        sym_params_best = inits[0].reshape(M, 2)

    # NM from best symmetric
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            res_nm = minimize(
                _objective_symmetric,
                sym_params_best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = _decode_central_symmetric(res_nm.x.reshape(M, 2))
            _update(pts_nm)
        except Exception:
            pass

    # NM from several diverse symmetric starts
    for x0 in inits[:15]:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        try:
            res = minimize(
                _objective_symmetric,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = _decode_central_symmetric(res.x.reshape(M, 2))
            _update(pts)
        except Exception:
            pass

    # ── Phase 3: Soft-objective L-BFGS in symmetric space ──
    if best_points is not None:
        bp = best_points.copy()
        angles_bp = np.arctan2(bp[:, 1], bp[:, 0])
        order_bp = np.argsort(angles_bp)
        bp_sorted = bp[order_bp]
        sym_params = bp_sorted[:M]

        remaining = deadline - time.time()
        if remaining > 10:
            try:
                res_lbfgs = minimize(
                    _objective_symmetric_soft,
                    sym_params.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                pts = _decode_central_symmetric(res_lbfgs.x.reshape(M, 2))
                _update(pts)
            except Exception:
                pass

    # ── Phase 4: Break symmetry – full 28D Cartesian polish ──
    if best_points is not None:
        cart_flat = best_points.flatten()

        # Powell in Cartesian
        remaining = deadline - time.time()
        if remaining > 10:
            try:
                res_powell = minimize(
                    _objective_hard,
                    cart_flat,
                    method='Powell',
                    options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-14}
                )
                _update(res_powell.x.reshape(N, 2))
            except Exception:
                pass

        # L-BFGS with soft objective in Cartesian
        remaining = deadline - time.time()
        if remaining > 8:
            try:
                res_lbg = minimize(
                    _objective_soft_cart,
                    cart_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                _update(res_lbg.x.reshape(N, 2))
            except Exception:
                pass

        # NM in Cartesian
        remaining = deadline - time.time()
        if remaining > 8:
            try:
                res_nm2 = minimize(
                    _objective_hard,
                    cart_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                _update(res_nm2.x.reshape(N, 2))
            except Exception:
                pass

    # ── Phase 5: Coordinate descent refinement ──
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 8:
            pts_cd = _coordinate_descent(best_points, deadline, max_rounds=30)
            _update(pts_cd)

    # ── Phase 6: Re-symmetrize and re-optimize (exploit symmetry after Cartesian drift) ──
    if best_points is not None:
        bp = best_points.copy()
        angles_bp = np.arctan2(bp[:, 1], bp[:, 0])
        order_bp = np.argsort(angles_bp)
        bp_sorted = bp[order_bp]
        sym_params2 = bp_sorted[:M]

        remaining = deadline - time.time()
        if remaining > 8:
            try:
                res_sym2 = minimize(
                    _objective_symmetric,
                    sym_params2.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts_sym2 = _decode_central_symmetric(res_sym2.x.reshape(M, 2))
                _update(pts_sym2)
            except Exception:
                pass

    # ── Phase 7: Final tight polish ──
    if best_points is not None:
        cart_flat = best_points.flatten()
        remaining = deadline - time.time()
        if remaining > 5:
            try:
                res_final = minimize(
                    _objective_soft_cart,
                    cart_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                _update(res_final.x.reshape(N, 2))
            except Exception:
                pass

        remaining = deadline - time.time()
        if remaining > 4:
            pts_cd2 = _coordinate_descent(best_points, deadline, max_rounds=20)
            _update(pts_cd2)

    # ── Fallback ──
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Normalize to unit hull area
    final_points = _normalize_unit_area(best_points)

    # Safety checks
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    if np.any(np.isnan(final_points)) or np.any(np.isinf(final_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
