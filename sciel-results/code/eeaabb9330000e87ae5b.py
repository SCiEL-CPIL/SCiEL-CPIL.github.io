import math
import time
import numpy as np
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
R = 1.0  # circumradius of unit hexagon
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0


def _time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN


def _axial_to_cartesian(axial_list, scale=1.0):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0) * scale
        y = 1.5 * r * scale
        pts.append((x, y))
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# SAT non-overlap gap for two unit hexagons with centers p,q and orientations th,th2
# gap = min over 12 axes of (p.n_a + q.n_a') - 2R   (>=0 means separated)
# ---------------------------------------------------------------------------
def _gap(p, q, th, th2):
    d = p - q
    # 6 axes of hexagon i, 6 of hexagon j
    a_i = th + np.arange(6) * (math.pi / 3.0)
    a_j = th2 + np.arange(6) * (math.pi / 3.0)
    # axis a_i: gap = d.n_i - 2R ; axis a_j: gap = -d.n_j - 2R
    n_i = np.column_stack([np.cos(a_i), np.sin(a_i)])
    n_j = np.column_stack([np.cos(a_j), np.sin(a_j)])
    g_i = d @ n_i.T - 2.0 * R
    g_j = -(d @ n_j.T) - 2.0 * R
    return float(np.min(np.concatenate([g_i, g_j])))


def _gap_grad(p, q, th, th2):
    d = p - q
    a_i = th + np.arange(6) * (math.pi / 3.0)
    a_j = th2 + np.arange(6) * (math.pi / 3.0)
    n_i = np.column_stack([np.cos(a_i), np.sin(a_i)])
    n_j = np.column_stack([np.cos(a_j), np.sin(a_j)])
    g_i = d @ n_i.T - 2.0 * R
    g_j = -(d @ n_j.T) - 2.0 * R
    allg = np.concatenate([g_i, g_j])
    m = np.argmin(allg)
    if m < 6:
        n = n_i[m]
        dp, dq = n, -n
        dth = -np.sum(d * np.array([-math.sin(a_i[m]), math.cos(a_i[m])]))
        dth2 = 0.0
    else:
        n = n_j[m - 6]
        dp, dq = -n, n
        dth = 0.0
        dth2 = np.sum(d * np.array([-math.sin(a_j[m - 6]), math.cos(a_j[m - 6])]))
    return float(np.min(allg)), dp, dq, dth, dth2


def _admm_solve(pts0, max_iter=3000, mu_init=0.4, tol=1e-12):
    n = len(pts0)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(pairs)
    p0 = pts0.copy()
    th = np.zeros(n)
    # build A (2m x 2n), b (2m)
    A = np.zeros((2 * m, 2 * n))
    b = np.zeros(2 * m)
    for k, (i, j) in enumerate(pairs):
        A[2 * k, 2 * i] = 1.0
        A[2 * k, 2 * j] = -1.0
        A[2 * k + 1, 2 * i] = 1.0
        A[2 * k + 1, 2 * j] = -1.0
    # precompute axis unit vectors (orientation-independent part: base hexagon axes)
    base_axes = np.column_stack([np.cos(np.arange(6) * (math.pi / 3.0)),
                                 np.sin(np.arange(6) * (math.pi / 3.0))])
    mu = mu_init
    lam = 0.0
    # x-step linear system: (I + A^T A / mu) x = x_prev + A^T (b - lam) / mu
    # solve once for M = I + A^T A / mu  (changes when mu changes)
    M = None
    prev_gap = float('inf')
    for it in range(max_iter):
        if not _time_ok():
            break
        # orientation step (subgradient ascent)
        if it % 2 == 0:
            step = 0.05 * (1.0 / (1.0 + it * 0.01))
            for i in range(n):
                gth = 0.0
                for k, (a, bidx) in enumerate(pairs):
                    if a == i:
                        g, dp, dq, dth, dth2 = _gap_grad(p0[a], p0[bidx], th[a], th[bidx])
                        gth += dth
                    elif bidx == i:
                        g, dp, dq, dth, dth2 = _gap_grad(p0[a], p0[bidx], th[a], th[bidx])
                        gth += dth2
                th[i] += step * gth
        # dual step: b = P(b + A x / mu)  where x = [p0; th-independent? no, x is positions]
        # note: orientation is NOT in x; we fix th and treat positions as x
        x = p0.reshape(-1)
        Ax = A @ x
        b = np.minimum(b + Ax / mu, 0.0)
        # x-step
        rhs = x + A.T @ (b - lam) / mu
        if M is None:
            M = np.eye(2 * n) + (A.T @ A) / mu
        try:
            xnew = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            xnew = np.linalg.lstsq(M, rhs, rcond=None)[0]
        p0 = xnew.reshape(n, 2)
        lam = b
        # check convergence
        if it % 20 == 0:
            gmin = min(_gap(p0[a], p0[bidx], th[a], th[bidx]) for a, bidx in pairs)
            if gmin < prev_gap - 1e-4:
                prev_gap = gmin
            if gmin > -1e-9 and it > 50:
                break
        # adapt mu
        if it % 100 == 0 and it > 0:
            gmin = min(_gap(p0[a], p0[bidx], th[a], th[bidx]) for a, bidx in pairs)
            if gmin < -1e-6:
                mu = min(mu * 1.5, 200.0)
                M = None
    return p0, th


def _orientation_lp(pts, beta):
    """Given fixed inner positions and outer orientation beta, find outer center + min S via LP."""
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    c = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            S = (res.x[2] + APOTHEM) / APOTHEM
            return S, (res.x[0], res.x[1])
    except Exception:
        pass
    return float('inf'), None


def _best_orientation(pts):
    best_S = float('inf')
    best_beta = 0.0
    best_c = None
    # coarse sweep
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, c = _orientation_lp(pts, beta)
        if S < best_S:
            best_S, best_beta, best_c = S, beta, c
    # refine
    try:
        def obj(b):
            S, _ = _orientation_lp(pts, b)
            return S
        res = minimize_scalar(obj, bounds=(max(0, best_beta - 0.03),
                                          min(math.pi / 6, best_beta + 0.03)),
                              method='bounded', options={'maxiter': 200, 'xatol': 1e-9})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = res.x
            best_S, best_c = _orientation_lp(pts, best_beta)
    except Exception:
        pass
    return best_S, best_beta, best_c


def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = [
        # flower (1 + 6 + 4)
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        # rows 4-3-4
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 3-5-3
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (4, 1), (0, 2), (1, 2), (2, 2)],
        # 5-3-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        # 4-4-3
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        # 3-4-4
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        # 2-3-3-3
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2),
         (2, 2), (0, 3), (1, 3), (2, 3)],
        # 3-3-3-2
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
         (1, 2), (2, 2), (0, 3), (1, 3)],
        # 4-3-4 variant
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        # 5-4-2
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1),
         (2, 1), (3, 1), (1, 2), (2, 2)],
        # 2-4-5
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2),
         (1, 2), (2, 2), (3, 2), (4, 2)],
        # 3-4-4 alt
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
    ]

    best_S = float('inf')
    best_result = None

    for idx, axial in enumerate(candidates):
        if not _time_ok():
            break
        # small expansion for strict disjointness
        pts0 = _axial_to_cartesian(axial, scale=1.0 + 1e-7)
        # center the point cloud at origin for better conditioning
        centroid = pts0.mean(axis=0)
        pts0 = pts0 - centroid

        # seed orientation via LP on the lattice
        S_seed, beta_seed, c_seed = _best_orientation(pts0)

        # ADMM with independent rotations
        try:
            p_sol, th_sol = _admm_solve(pts0, max_iter=1500, mu_init=0.5)
        except Exception:
            p_sol, th_sol = pts0, np.zeros(len(pts0))

        # verify non-overlap; if violated, fall back to lattice (no independent rotation)
        overlap = False
        for i in range(len(p_sol)):
            for j in range(i + 1, len(p_sol)):
                g = _gap(p_sol[i], p_sol[j], th_sol[i], th_sol[j])
                if g < -1e-6:
                    overlap = True
                    break
            if overlap:
                break

        if overlap:
            # use lattice positions with shared orientation
            p_use = pts0
            th_use = np.full(len(pts0), beta_seed)
        else:
            p_use = p_sol
            th_use = th_sol

        # find best outer orientation for the solved configuration
        S_out, beta_out, c_out = _best_orientation(p_use)
        if S_out < best_S and c_out is not None:
            best_S = S_out
            best_beta = beta_out
            best_result = (p_use, th_use, c_out, best_beta)

    if best_result is None or not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    p_use, th_use, c_out, best_beta = best_result
    inner = []
    for k in range(len(p_use)):
        inner.append([float(p_use[k, 0]), float(p_use[k, 1]),
                      float(math.degrees(th_use[k]))])
    center = [float(c_out[0]), float(c_out[1])]
    side = float(best_S)
    rotation = float(math.degrees(best_beta))
    return inner, center, side, rotation
