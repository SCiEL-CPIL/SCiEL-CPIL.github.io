import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

def _compute_areas(points):
    """Compute all triangle areas efficiently using vectorized cross products."""
    p = points
    # C[i,j] = p[i][0]*p[j][1] - p[i][1]*p[j][0]
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.015):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s = np.sum(np.exp(shifted / temperature))
    soft_min = m - temperature * np.log(s)
    return soft_min / hull_area

def _two_temp_soft_min(points, temp1=0.008, temp2=0.04):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    s1 = np.sum(np.exp(shifted / temp1))
    soft1 = m - temp1 * np.log(s1)
    s2 = np.sum(np.exp(shifted / temp2))
    soft2 = m - temp2 * np.log(s2)
    return (soft1 * soft2) ** 0.5 / hull_area

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    return pts_out

def _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    p1 = np.column_stack((np.cos(a1), np.sin(a1)))
    if n2 > 0:
        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phi2
        p2 = np.column_stack((r2 * np.cos(a2), r2 * np.sin(a2)))
    else:
        p2 = np.empty((0, 2))
    if n3 > 0:
        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phi3
        p3 = np.column_stack((r3 * np.cos(a3), r3 * np.sin(a3)))
    else:
        p3 = np.empty((0, 2))
    return np.vstack((p1, p2, p3))

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r_in, phi_rel = params
        if r_in < 0.0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        return -_calculate_score(pts)

    r_starts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    phi_starts = [0.0]
    if n_in > 0:
        phi_starts += [np.pi / n_in, np.pi / (2.0 * n_in), 2.0 * np.pi / n_in,
                       np.pi / (3.0 * n_in), np.pi / (4.0 * n_in)]
    if n_out > 0:
        phi_starts += [np.pi / n_out, np.pi / (2.0 * n_out)]

    for r0 in r_starts:
        for phi0 in phi_starts:
            if time.time() - start_time > time_limit:
                return best_score, best_params
            try:
                res = minimize(objective, [r0, phi0], method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    return best_score, best_params

def _optimize_three_ring(n1, n2, n3, time_limit=8.0):
    start_time = time.time()
    best_score = 0.0
    best_params = None

    def objective(params):
        r2, r3, phi2, phi3 = params
        if r2 < 0.0 or r2 > 1.5 or r3 < 0.0 or r3 > 1.5:
            return 1e6
        pts = _generate_three_ring(n1, n2, n3, r2, r3, phi2, phi3)
        return -_calculate_score(pts)

    r2_starts = [0.3, 0.5, 0.7, 0.9]
    r3_starts = [0.2, 0.4, 0.6, 0.8]
    phi2_starts = [0.0, np.pi / n2 if n2 > 0 else 0.0, np.pi / (2.0 * n2) if n2 > 0 else 0.0]
    phi3_starts = [0.0, np.pi / n3 if n3 > 0 else 0.0, np.pi / (2.0 * n3) if n3 > 0 else 0.0]

    count = 0
    for r2 in r2_starts:
        for r3 in r3_starts:
            for phi2 in phi2_starts:
                for phi3 in phi3_starts:
                    if time.time() - start_time > time_limit or count >= 18:
                        return best_score, best_params
                    count += 1
                    try:
                        res = minimize(objective, [r2, r3, phi2, phi3], method='Nelder-Mead',
                                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 400, 'disp': False})
                        if -res.fun > best_score:
                            best_score = -res.fun
                            best_params = res.x.copy()
                    except Exception:
                        pass
    return best_score, best_params

def _generate_single_ring(angles):
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _optimize_single_ring(time_limit=6.0):
    start_time = time.time()
    best_score = 0.0
    best_angles = None

    def score_from_angles(a):
        pts = _generate_single_ring(a)
        return _calculate_score(pts)

    def neg_score(a):
        return -score_from_angles(a)

    base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    s = score_from_angles(base)
    if s > best_score:
        best_score, best_angles = s, base.copy()

    rng = np.random.default_rng(7)
    for trial in range(6):
        if time.time() - start_time > time_limit:
            break
        a0 = base + rng.uniform(-0.15, 0.15, size=N)
        try:
            res = minimize(neg_score, a0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 600, 'disp': False})
            s = score_from_angles(res.x)
            if s > best_score:
                best_score, best_angles = s, res.x.copy()
        except Exception:
            pass
    return best_score, best_angles

def _coordinate_descent_polish(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(4):
        if time.time() - start_time > time_limit:
            break
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            old_pt = p[i].copy()

            def obj_2d(x):
                p[i] = x
                return -_calculate_score(p)

            try:
                res = minimize(obj_2d, old_pt, method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 300, 'disp': False})
                p[i] = res.x
                s_new = _calculate_score(p)
                if s_new <= s - 1e-12:
                    p[i] = old_pt
            except Exception:
                p[i] = old_pt
        s = _calculate_score(p)
    return p, s

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    def neg_two_temp(x):
        return -_two_temp_soft_min(x.reshape(N, 2))

    def neg_soft(x):
        return -_soft_min_score(x.reshape(N, 2))

    def neg_hard(x):
        return -_calculate_score(x.reshape(N, 2))

    # L-BFGS-B on two-temperature smooth surrogate
    try:
        res = minimize(neg_two_temp, p.flatten(), method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11})
        np2 = res.x.reshape(N, 2)
        s2 = _calculate_score(np2)
        if s2 > s:
            p, s = np2, s2
    except Exception:
        pass

    # L-BFGS-B on single-temperature smooth surrogate
    if time.time() - start_time < time_limit * 0.7:
        try:
            res = minimize(neg_soft, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # L-BFGS-B on the true (piecewise) objective
    if time.time() - start_time < time_limit * 0.5:
        try:
            res = minimize(neg_hard, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Coordinate descent polish
    if time.time() - start_time < time_limit * 0.4:
        cd_limit = time_limit - (time.time() - start_time) - 0.5
        if cd_limit > 1.0:
            try:
                p, s = _coordinate_descent_polish(p, time_limit=cd_limit)
            except Exception:
                pass

    # Nelder-Mead fallback
    if time.time() - start_time < time_limit * 0.3:
        nm_limit = time_limit - (time.time() - start_time) - 0.5
        if nm_limit > 1.0:
            try:
                res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-10, 'maxiter': 600, 'disp': False})
                np2 = res.x.reshape(N, 2)
                s2 = _calculate_score(np2)
                if s2 > s:
                    p, s = np2, s2
            except Exception:
                pass

    return p, s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Single-ring angular-packing family
    if time.time() < deadline - 25:
        s, ang = _optimize_single_ring(time_limit=5.0)
        if ang is not None:
            pts = _generate_single_ring(ang)
            polished, ps = _polish_points(pts, time_limit=3.0)
            try_update(ps, polished)

    # 2. Two-ring sweep
    splits = [(14, 0), (13, 1), (12, 2), (11, 3), (10, 4), (9, 5), (8, 6), (7, 7)]
    for n_out, n_in in splits:
        if time.time() > deadline - 20:
            break
        if n_in == 0:
            pts = _generate_two_ring(n_out, 0, 0.0, 0.0)
            polished, ps = _polish_points(pts, time_limit=2.0)
            try_update(ps, polished)
            continue
        if n_out == 7 and n_in == 7:
            tl = 10.0
        else:
            tl = 5.0
        score, params = _optimize_two_ring(n_out, n_in, time_limit=tl)
        if params is not None:
            pts = _generate_two_ring(n_out, n_in, params[0], params[1])
            polished, ps = _polish_points(pts, time_limit=4.0)
            try_update(ps, polished)

    # 3. Three-ring exploration (new structural family)
    three_splits = [(10, 2, 2), (9, 3, 2), (8, 4, 2), (7, 4, 3), (8, 3, 3), (7, 3, 4)]
    for n1, n2, n3 in three_splits:
        if time.time() > deadline - 15:
            break
        score, params = _optimize_three_ring(n1, n2, n3, time_limit=6.0)
        if params is not None:
            pts = _generate_three_ring(n1, n2, n3, params[0], params[1], params[2], params[3])
            polished, ps = _polish_points(pts, time_limit=4.0)
            try_update(ps, polished)

    # 4. Extra polish passes on the best result if time remains
    if best_points is not None and time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 3:
            polished, ps = _polish_points(best_points, time_limit=remaining)
            try_update(ps, polished)

    # 5. Final coordinate descent + Nelder-Mead on the very best
    if best_points is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            try:
                p_cd, s_cd = _coordinate_descent_polish(best_points, time_limit=remaining * 0.5)
                try_update(s_cd, p_cd)
            except Exception:
                pass
            if time.time() < deadline - 2:
                remaining2 = deadline - time.time() - 1
                if remaining2 > 2:
                    try:
                        p2, s2 = _polish_points(best_points, time_limit=remaining2)
                        try_update(s2, p2)
                    except Exception:
                        pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
