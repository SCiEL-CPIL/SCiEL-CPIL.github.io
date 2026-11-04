"""
Heilbronn convex regions n=14: maximize min triangle area / hull area.
Combined strategy: Fourier-radius parameterization for global search +
coordinate-descent polish with efficient area computation.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# Precompute triangle indices
_i_idx, _j_idx, _k_idx = [], [], []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _i_idx.append(i)
            _j_idx.append(j)
            _k_idx.append(k)
_I = np.array(_i_idx)
_J = np.array(_j_idx)
_K = np.array(_k_idx)

# Precompute base angles for regular 14-gon
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS_A = np.cos(_BASE_ANGLES)
_SIN_A = np.sin(_BASE_ANGLES)


def _all_triangle_areas(pts):
    """Return array of all C(14,3)=364 triangle areas using vectorized cross products."""
    # Efficient method: C[i,j] = p[i][0]*p[j][1] - p[i][1]*p[j][0]
    C = pts[:, 0:1] * pts[:, 1:2].T - pts[:, 1:2] * pts[:, 0:1].T
    vals = C[_J, _K] + C[_I, _J] - C[_I, _K]
    return 0.5 * np.abs(vals)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1e-10


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 0.0
    return np.min(_all_triangle_areas(pts)) / ha


def _soft_min(areas, beta):
    m = np.min(areas)
    shifted = areas - m
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))


def _fourier_to_points(c, max_mode=4):
    """
    c = [a1, b1, a2, b2, ...] up to max_mode.
    radius_i = 1 + sum_{k=1}^{max_mode} (a_k cos(k*theta_i) + b_k sin(k*theta_i))
    """
    radii = np.ones(N)
    for k in range(1, max_mode + 1):
        a = c[2 * (k - 1)]
        b = c[2 * (k - 1) + 1]
        radii += a * np.cos(k * _BASE_ANGLES) + b * np.sin(k * _BASE_ANGLES)
    radii = np.clip(radii, 0.1, 2.0)
    return radii[:, None] * np.column_stack([_COS_A, _SIN_A])


def _neg_score_fourier(c, max_mode=4):
    pts = _fourier_to_points(c, max_mode)
    return -_score(pts)


def _neg_soft_fourier(c, beta, max_mode=4):
    pts = _fourier_to_points(c, max_mode)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _neg_score_full(x):
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _neg_soft_full(x, beta):
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _coordinate_descent_polish(points, deadline, label=""):
    """
    Coordinate descent: optimize each point in turn using Nelder-Mead.
    This avoids repeated hull computations within a single point optimization.
    """
    p = points.copy()
    s = _score(p)
    start_time = time.time()

    # Multiple passes of coordinate descent
    for iteration in range(6):
        if time.time() > deadline - 10:
            break
        for i in range(N):
            if time.time() > deadline - 10:
                break
            old_pt = p[i].copy()

            def obj_2d(x):
                p[i] = x
                return -_score(p)

            try:
                res = minimize(obj_2d, old_pt, method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9,
                                        'maxiter': 200, 'disp': False})
                new_s = _score(p)
                if new_s > s + 1e-12:
                    s = new_s
                else:
                    p[i] = old_pt
            except Exception:
                p[i] = old_pt

    return p, s


def _polish(pts, deadline, label=""):
    """Multi-stage polish: coordinate descent + L-BFGS-B + Powell."""
    best_pts = pts.copy()
    best_s = _score(best_pts)

    # Stage 1: Coordinate descent (efficient, no repeated hull in inner loop)
    if time.time() < deadline - 15:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, deadline - 10, label + "_cd")
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts.copy()

    # Stage 2: Continuation L-BFGS-B on soft-min
    betas = [30.0, 100.0, 400.0]
    x = best_pts.flatten()
    for beta in betas:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(_neg_soft_full, x, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
                x = best_pts.flatten()
        except Exception:
            pass

    # Stage 3: Final Powell on hard objective
    if time.time() < deadline - 4:
        try:
            res = minimize(_neg_score_full, x, method='Powell',
                           options={'maxiter': 200, 'xtol': 1e-10, 'ftol': 1e-12})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass

    # Stage 4: Final coordinate descent pass
    if time.time() < deadline - 3:
        cd_pts, cd_s = _coordinate_descent_polish(best_pts, deadline - 2, label + "_cd2")
        if cd_s > best_s:
            best_s = cd_s
            best_pts = cd_pts.copy()

    return best_pts, best_s


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    best_pts = None
    best_score = -np.inf

    def update(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular 14-gon baseline
    reg_pts = np.column_stack([_COS_A, _SIN_A])
    update(reg_pts)

    # 2. Fourier-radius DE search (8-dim: modes 1-4)
    if time.time() < deadline - 55:
        bounds = [(-0.3, 0.3)] * 8
        try:
            res = differential_evolution(_neg_score_fourier, bounds, args=(4,),
                                         seed=7, maxiter=250, tol=1e-8,
                                         polish=True, popsize=15)
            if res.fun < 0:
                pts = _fourier_to_points(res.x, 4)
                update(pts)
        except Exception:
            pass

    # 3. Second DE with more modes (12-dim: modes 1-6)
    if time.time() < deadline - 45:
        bounds6 = [(-0.2, 0.2)] * 12
        try:
            res = differential_evolution(_neg_score_fourier, bounds6, args=(6,),
                                         seed=13, maxiter=180, tol=1e-7,
                                         polish=True, popsize=12)
            if res.fun < 0:
                pts = _fourier_to_points(res.x, 6)
                update(pts)
        except Exception:
            pass

    # 4. Two-ring DE (inner 7 + outer 7) as alternative family
    if time.time() < deadline - 35:
        def _two_ring_pts(x):
            r = x[0]
            thi = x[1:8]
            tho = x[8:15]
            ip = np.column_stack([r * np.cos(thi), r * np.sin(thi)])
            op = np.column_stack([np.cos(tho), np.sin(tho)])
            return np.vstack([ip, op])

        def _neg_two_ring(x):
            return -_score(_two_ring_pts(x))

        b2 = [(0.2, 0.8)] + [(0, 2 * np.pi)] * 14
        try:
            res = differential_evolution(_neg_two_ring, b2, seed=21,
                                         maxiter=120, tol=1e-6, polish=True,
                                         popsize=12)
            if res.fun < 0:
                update(_two_ring_pts(res.x))
        except Exception:
            pass

    # 5. Polish the best found so far
    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_score = _polish(best_pts, deadline - 5, "main")
        update(best_pts)

    # 6. Random restarts + polish if time remains
    rng = np.random.RandomState(42)
    for restart in range(4):
        if time.time() > deadline - 25:
            break
        # Perturb regular polygon
        noise = rng.normal(0, 0.12, size=(N, 2))
        init = reg_pts + noise
        update(init)
        if time.time() < deadline - 15:
            polished, s_pol = _polish(init, deadline - 10, f"r{restart}")
            update(polished)

    # 7. Final global polish on the very best
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _polish(best_pts, deadline, "final")
        update(best_pts)

    if best_pts is None:
        return reg_pts
    return best_pts
