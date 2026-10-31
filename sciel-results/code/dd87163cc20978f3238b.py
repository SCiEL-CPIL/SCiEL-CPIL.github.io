import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _generate_grid_points():
    coords = np.array([
        [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5],
        [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
        [-1.5, 0.5],  [-0.5, 0.5],  [0.5, 0.5],  [1.5, 0.5],
        [-1.5, 1.5],  [-0.5, 1.5],  [0.5, 1.5],  [1.5, 1.5]
    ])
    coords = coords[:-1]
    return coords


def _apply_affine_transform(points, a11, a12, a21, a22, tx, ty):
    A = np.array([[a11, a12], [a21, a22]])
    t = np.array([tx, ty])
    return points @ A.T + t


def _optimize_affine_grid(deadline):
    base_points = _generate_grid_points()
    
    def obj(p):
        a11, a12, a21, a22, tx, ty = p
        pts = _apply_affine_transform(base_points, a11, a12, a21, a22, tx, ty)
        return _neg_score(pts)

    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 10:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 5:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _apply_affine_transform(base_points, *best_p)
    return pts, best_s


def _optimize_from_points(points, deadline, max_iters_powell=400):
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                x0,
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def _optimize_two_ring(n_out, r_inner, angle_offset, deadline):
    n_in = N - n_out
    p0 = np.zeros(2 + 2 * N)
    p0[0] = r_inner
    p0[1] = angle_offset

    def obj(p):
        r_in = float(p[0])
        a_off = float(p[1])
        rp = p[2:2 + N]
        ap = p[2 + N:2 + 2 * N]
        a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + ap[:n_out]
        a_in = np.linspace(a_off, 2 * np.pi + a_off, n_in, endpoint=False) + ap[n_out:]
        r_out = np.ones(n_out) * (1.0 + rp[:n_out])
        r_in = r_in * (1.0 + rp[n_out:])
        outer = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
        inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
        pts = np.vstack([outer, inner])
        return _neg_score(pts)

    x0 = p0
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    r_in = float(best_p[0])
    a_off = float(best_p[1])
    rp = best_p[2:2 + N]
    ap = best_p[2 + N:2 + 2 * N]
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False) + ap[:n_out]
    a_in = np.linspace(a_off, 2 * np.pi + a_off, n_in, endpoint=False) + ap[n_out:]
    r_out = np.ones(n_out) * (1.0 + rp[:n_out])
    r_in = r_in * (1.0 + rp[n_out:])
    outer = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    pts = np.vstack([outer, inner])
    return pts, best_s


def _optimize_three_ring(n1, n2, n3, r2, r3, off2, off3, deadline):
    def obj(p):
        rr2, rr3, o2, o3 = p[0], p[1], p[2], p[3]
        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        a2 = np.linspace(o2, 2 * np.pi + o2, n2, endpoint=False)
        a3 = np.linspace(o3, 2 * np.pi + o3, n3, endpoint=False)
        pts = np.vstack([
            np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([rr2 * np.cos(a2), rr2 * np.sin(a2)]),
            np.column_stack([rr3 * np.cos(a3), rr3 * np.sin(a3)]),
        ])
        return _neg_score(pts)

    x0 = np.array([r2, r3, off2, off3], dtype=float)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 250, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    rr2, rr3, o2, o3 = float(best_p[0]), float(best_p[1]), float(best_p[2]), float(best_p[3])
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(o2, 2 * np.pi + o2, n2, endpoint=False)
    a3 = np.linspace(o3, 2 * np.pi + o3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([rr2 * np.cos(a2), rr2 * np.sin(a2)]),
        np.column_stack([rr3 * np.cos(a3), rr3 * np.sin(a3)]),
    ])
    return pts, best_s


def _optimize_regular_perturbed(deadline):
    def obj(p):
        rp = p[:N]
        ap = p[N:2 * N]
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + ap
        radii = 1.0 + rp
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return _neg_score(pts)

    x0 = np.zeros(2 * N)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    rp = best_p[:N]
    ap = best_p[N:2 * N]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + ap
    radii = 1.0 + rp
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts, best_s


def _radial_to_cartesian(params):
    radii = np.abs(params[:N])
    angles = params[N:]
    order = np.argsort(angles)
    radii_sorted = radii[order]
    angles_sorted = angles[order]
    angles_sorted = angles_sorted - angles_sorted[0]
    angles_sorted = np.sort(angles_sorted)
    pts = np.column_stack([radii_sorted * np.cos(angles_sorted),
                           radii_sorted * np.sin(angles_sorted)])
    return pts


def _cartesian_to_radial(points):
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    order = np.argsort(angles)
    return np.concatenate([radii[order], angles[order]])


def _neg_score_radial(params):
    pts = _radial_to_cartesian(params)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_radial(params, temperature):
    pts = _radial_to_cartesian(params)
    s = _soft_score(pts, temperature)
    if s <= 0:
        return 1e10
    return -s


def _annealed_soft_minimize(points, deadline, max_temp=0.05, min_temp=1e-5, n_steps=8):
    best = points.copy()
    best_s = _score(best)
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='Powell',
                    options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

    return best, best_s


def _penalty_objective(x, mu):
    t = x[0]
    pts = x[1:].reshape(N, 2)
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    return -t / hull_a + mu * penalty


def _penalty_optimize(points, deadline, mu_start=10.0, mu_end=1e6, n_mu_steps=6,
                      max_iter_per_mu=150):
    best = points.copy()
    best_s = _score(best)
    
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.9
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:] = best.flatten()
    
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                lambda x: _penalty_objective(x, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            x0 = res.x
            cand_pts = x0[1:].reshape(N, 2)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s


def _radial_optimize(points, deadline):
    best = points.copy()
    best_s = _score(best)
    p0 = _cartesian_to_radial(best)
    
    if time.time() < deadline - 6:
        try:
            res = minimize(
                _neg_score_radial,
                p0,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8},
            )
            cand = _radial_to_cartesian(res.x)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(
                _neg_score_radial,
                _cartesian_to_radial(best),
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-11, 'fatol': 1e-11, 'adaptive': True},
            )
            cand = _radial_to_cartesian(res.x)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Seed candidates ----------
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(np.column_stack([np.cos(ang), np.sin(ang)]))
    
    grid_pts = _generate_grid_points()
    _update(grid_pts)

    # ---------- Affine Grid Optimization ----------
    if time.time() < deadline - 15:
        try:
            pts, s = _optimize_affine_grid(deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Two-ring models ----------
    two_ring_configs = []
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out), np.pi / n_in]:
                two_ring_configs.append((n_out, r_in, a_off))

    for (n_out, r_in, a_off) in two_ring_configs:
        if time.time() > deadline - 10:
            break
        try:
            pts, s = _optimize_two_ring(n_out, r_in, a_off, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Three-ring models ----------
    three_ring_configs = [
        (5, 4, 4, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.6, 0.3, 0.0, 0.0),
        (7, 4, 2, 0.6, 0.3, 0.0, 0.0),
        (6, 5, 2, 0.6, 0.3, 0.0, 0.0),
        (7, 3, 3, 0.6, 0.3, 0.0, 0.0),
        (5, 5, 3, 0.6, 0.3, 0.0, 0.0),
        (6, 4, 3, 0.5, 0.25, np.pi / 6, 0.0),
        (7, 4, 2, 0.5, 0.25, np.pi / 7, 0.0),
    ]
    for (n1, n2, n3, r2, r3, o2, o3) in three_ring_configs:
        if time.time() > deadline - 8:
            break
        try:
            pts, s = _optimize_three_ring(n1, n2, n3, r2, r3, o2, o3, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Regular 13-gon with perturbations ----------
    if time.time() < deadline - 10:
        try:
            pts, s = _optimize_regular_perturbed(deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Annealed Soft Min (NEW) ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Penalty-based optimization (NEW) ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _penalty_optimize(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Radial coordinate optimization (NEW) ----------
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _radial_optimize(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate polish ----------
    if best_points is not None and time.time() < deadline - 4:
        try:
            pts, s = _optimize_from_points(best_points, deadline, max_iters_powell=300)
            _update(pts)
        except Exception:
            pass

    # ---------- Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 1.5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_points
