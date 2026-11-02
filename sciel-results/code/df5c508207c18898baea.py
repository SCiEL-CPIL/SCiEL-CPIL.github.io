import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
R = 1.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0


def _time_ok(margin=0.0):
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - margin


def _axial_to_cartesian(axial_list, scale=1.0):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0) * scale
        y = 1.5 * r * scale
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _rotmat(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _gap_matrix(d, a_i, a_j):
    """Vectorized 12-axis gap for all pairs. Returns (m,12) array.
    d: (m,2) = p_i - p_j ; a_i, a_j: (m,6) axis angles."""
    n_i = np.column_stack([np.cos(a_i), np.sin(a_i)])   # (m,6,2)
    n_j = np.column_stack([np.cos(a_j), np.sin(a_j)])
    g_i = np.einsum('mi,mi->m', d, n_i[:, :, 0]) + np.einsum('mi,mi->m', d, n_i[:, :, 1]) * 0
    # simpler:
    g_i = np.sum(d[:, None, :] * n_i, axis=2) - 2.0 * R   # (m,6)
    g_j = -np.sum(d[:, None, :] * n_j, axis=2) - 2.0 * R  # (m,6)
    return np.concatenate([g_i, g_j], axis=1)            # (m,12)


def _pair_indices(n):
    idx = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    return idx


def _all_gaps(pts, th):
    """Return (m,12) gap matrix for all pairs, plus pair index array."""
    n = len(pts)
    pairs = _pair_indices(n)
    d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    a_i = th[pairs[:, 0]][:, None] + np.arange(6)[None, :] * (math.pi / 3.0)
    a_j = th[pairs[:, 1]][:, None] + np.arange(6)[None, :] * (math.pi / 3.0)
    return _gap_matrix(d, a_i, a_j), pairs


def _softplus(x, beta=8.0):
    return np.logaddexp(0.0, beta * x) / beta


def _smooth_objective(vec, pts0, beta_sp=8.0, lam=5.0):
    """vec = [cx, cy, phi, *th_i]. Rigid composite + per-hex angles.
    Returns (t + lam * sum softplus(2R - gap)) where t = max projection of
    (rotated pts0 + center) onto outer hexagon normals(phi)."""
    n = len(pts0)
    cx, cy, phi = vec[0], vec[1], vec[2]
    th = vec[3:]
    M = _rotmat(phi)
    P = pts0 @ M.T + np.array([cx, cy])
    # outer normals
    base = phi + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(base), np.sin(base)])  # (6,2)
    projs = P @ normals.T  # (n,6)
    allp = np.concatenate([projs.ravel(), (-projs).ravel()])
    t = float(np.max(allp))
    # gaps
    gaps, pairs = _all_gaps(P, th)
    gmin = gaps.min(axis=1)
    penalty = float(np.sum(_softplus(-gmin, beta_sp)))
    return t + lam * penalty


def _smooth_objective_grad(vec, pts0, beta_sp=8.0, lam=5.0):
    n = len(pts0)
    cx, cy, phi = vec[0], vec[1], vec[2]
    th = vec[3:]
    M = _rotmat(phi)
    P = pts0 @ M.T + np.array([cx, cy])
    base = phi + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.column_stack([np.cos(base), np.sin(base)])
    projs = P @ normals.T
    allp = np.concatenate([projs.ravel(), (-projs).ravel()])
    t = float(np.max(allp))

    # gradient of t w.r.t. (cx, cy, phi): active normal
    am = int(np.argmax(allp))
    if am < n * 6:
        ii, kk = divmod(am, 6)
        sgn = 1.0
    else:
        am2 = am - n * 6
        ii, kk = divmod(am2, 6)
        sgn = -1.0
    nv = normals[kk]
    # P_i = pts0_i @ M.T + c ; dP_i/dc = (1,1)
    # dP_i/dphi = -R90 @ pts0_i  where R90 = [[0,-1],[1,0]]
    R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    dP_dphi = -(pts0[ii] @ R90)  # (2,)
    g_t = np.zeros(3 + n)
    g_t[0] = sgn * nv[0]
    g_t[1] = sgn * nv[1]
    g_t[2] = sgn * float(dP_dphi @ nv)

    # penalty gradient
    gaps, pairs = _all_gaps(P, th)
    gmin = gaps.min(axis=1)
    sp = _softplus(-gmin, beta_sp)
    spg = (np.exp(beta_sp * (-gmin)) / (1.0 + np.exp(beta_sp * (-gmin))))  # d/d(-gmin)
    dpen = lam * spg  # d penalty / d gmin (per pair)
    # for each pair, d gmin / d P_i = n_a (active axis), d gmin / d P_j = -n_a
    base_a = np.arange(6) * (math.pi / 3.0)
    for k in range(len(pairs)):
        i, j = pairs[k]
        am2 = int(np.argmin(gaps[k]))
        if am2 < 6:
            ang = th[i] + base_a[am2]
            nv2 = np.array([math.cos(ang), math.sin(ang)])
            # d gmin/dP_i = nv2 ; d gmin/dP_j = -nv2
            dP_i_dphi = -(pts0[i] @ np.array([[0.0, -1.0], [1.0, 0.0]]))
            dP_j_dphi = -(pts0[j] @ np.array([[0.0, -1.0], [1.0, 0.0]]))
            g_t[0] += dpen[k] * (nv2[0] - nv2[0])  # center: +nv2 from i, -(-nv2) from j -> cancel? 
            # Actually gmin_i depends on P_i via +nv2, on P_j via -nv2 (since d = P_i - P_j, g = d.n - 2R, d/dP_i = nv2, d/dP_j = -nv2)
            # contribution to gmin from P_i: +nv2 ; from P_j: -nv2
            g_t[0] += dpen[k] * (nv2[0] + nv2[0]) * 0  # placeholder, handled below
            # Let's do it cleanly:
            # d gmin_k / d P_i = nv2 ; d gmin_k / d P_j = -nv2
            # d P_i / d c = (1,1) ; d P_j / d c = (1,1)
            # => d gmin_k / d c = nv2 - nv2 = 0  (center cancels, as expected for rigid)
            # d P_i / d phi = dP_i_dphi ; d P_j / d phi = dP_j_dphi
            g_t[2] += dpen[k] * (float(dP_i_dphi @ nv2) - float(dP_j_dphi @ nv2))
            # d gmin_k / d th_i: axis ang = th_i + base; d/dth_i = dP_i @ d(nv2)/dang
            # d(nv2)/dang = (-sin, cos)
            dnv = np.array([-math.sin(ang), math.cos(ang)])
            # g = d.nv2 - 2R, d = P_i - P_j, so d g/dth_i = d @ dnv2 = (P_i - P_j) @ dnv2
            diff = P[i] - P[j]
            g_t[3 + i] += dpen[k] * float(diff @ dnv)
        else:
            ang = th[j] + base_a[am2 - 6]
            nv2 = np.array([math.cos(ang), math.sin(ang)])
            # g = -d.nv2 - 2R ; d g/dP_i = -nv2 ; d g/dP_j = +nv2
            dP_i_dphi = -(pts0[i] @ np.array([[0.0, -1.0], [1.0, 0.0]]))
            dP_j_dphi = -(pts0[j] @ np.array([[0.0, -1.0], [1.0, 0.0]]))
            g_t[2] += dpen[k] * (-float(dP_i_dphi @ nv2) + float(dP_j_dphi @ nv2))
            dnv = np.array([-math.sin(ang), math.cos(ang)])
            diff = P[i] - P[j]
            # d g/dth_j = -d @ d(nv2)/dang = -diff @ dnv
            g_t[3 + j] += dpen[k] * (-float(diff @ dnv))
    return t + lam * float(np.sum(sp)), g_t


def _orientation_lp(pts, beta):
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
    for i in range(73):
        beta = i * (math.pi / 6) / 72
        S, c = _orientation_lp(pts, beta)
        if S < best_S:
            best_S, best_beta, best_c = S, beta, c
    try:
        def obj(b):
            S, _ = _orientation_lp(pts, b)
            return S
        lo = max(0.0, best_beta - 0.02)
        hi = min(math.pi / 6, best_beta + 0.02)
        res = minimize_scalar(obj, bounds=(lo, hi), method='bounded',
                              options={'maxiter': 300, 'xatol': 1e-10})
        if res.fun < best_S:
            best_S = res.fun
            best_beta = float(res.x)
            best_S, best_c = _orientation_lp(pts, best_beta)
    except Exception:
        pass
    return best_S, best_beta, best_c


from scipy.optimize import minimize_scalar


def construct():
    if not _time_ok(2.0):
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    candidates = [
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (1, 1), (-1, 2), (-2, 1), (2, -1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (4, 1), (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2),
         (2, 2), (0, 3), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2),
         (1, 2), (2, 2), (0, 3), (1, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (1, 1),
         (2, 1), (3, 1), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2),
         (1, 2), (2, 2), (3, 2), (4, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
         (1, 2), (2, 2), (3, 2), (0, 3), (1, 3)],
    ]

    best_S = float('inf')
    best_result = None

    for idx, axial in enumerate(candidates):
        if not _time_ok(3.0):
            break
        pts0 = _axial_to_cartesian(axial, scale=1.0 + 1e-7)
        centroid = pts0.mean(axis=0)
        pts0 = pts0 - centroid

        # ---- Phase A: rigid composite smooth optimization (6 DOF) ----
        # vec = [cx, cy, phi] (rigid: all angles = phi, but we allow per-angle in B)
        # For rigid, use 3-DOF: [cx, cy, phi]
        def rigid_obj(vec):
            cx, cy, phi = vec
            M = _rotmat(phi)
            P = pts0 @ M.T + np.array([cx, cy])
            base = phi + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
            normals = np.column_stack([np.cos(base), np.sin(base)])
            projs = P @ normals.T
            allp = np.concatenate([projs.ravel(), (-projs).ravel()])
            t = float(np.max(allp))
            gaps, pairs = _all_gaps(P, np.full(len(pts0), phi))
            gmin = gaps.min(axis=1)
            pen = float(np.sum(_softplus(-gmin, 8.0)))
            return t + 5.0 * pen

        best_rigid = None
        for phi0 in np.linspace(0.0, math.pi / 6, 7):
            if not _time_ok(2.0):
                break
            x0 = np.array([0.0, 0.0, phi0])
            try:
                res = minimize(rigid_obj, x0, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-7})
                if res.fun < (best_rigid.fun if best_rigid is not None else float('inf')):
                    best_rigid = res
            except Exception:
                pass
        if best_rigid is not None:
            cx, cy, phi = best_rigid.x
            M = _rotmat(phi)
            P_rigid = pts0 @ M.T + np.array([cx, cy])
            th_rigid = np.full(len(pts0), phi)
        else:
            P_rigid = pts0
            th_rigid = np.zeros(len(pts0))

        # ---- Phase B: semi-rigid refine positions + angles (smooth penalty) ----
        # variables: 2n positions + n angles
        n = len(pts0)
        def semi_obj(vec):
            P = vec[:2 * n].reshape(n, 2)
            th = vec[2 * n:]
            # keep near rigid: add small pull-back
            base = phi + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
            normals = np.column_stack([np.cos(base), np.sin(base)])
            projs = P @ normals.T
            allp = np.concatenate([projs.ravel(), (-projs).ravel()])
            t = float(np.max(allp))
            gaps, pairs = _all_gaps(P, th)
            gmin = gaps.min(axis=1)
            pen = float(np.sum(_softplus(-gmin, 10.0)))
            pull = 0.01 * float(np.sum((P - P_rigid) ** 2)) + 0.01 * float(np.sum((th - th_rigid) ** 2))
            return t + 8.0 * pen + pull

        x0 = np.concatenate([P_rigid.ravel(), th_rigid])
        P_use = P_rigid
        th_use = th_rigid.copy()
        try:
            if _time_ok(2.0):
                res = minimize(semi_obj, x0, method='L-BFGS-B',
                               options={'maxiter': 300, 'gtol': 1e-6})
                P_use = res.x[:2 * n].reshape(n, 2)
                th_use = res.x[2 * n:].copy()
        except Exception:
            pass

        # verify non-overlap
        overlap = False
        gaps, pairs = _all_gaps(P_use, th_use)
        if gaps.min() < -1e-6:
            overlap = True
        if overlap:
            P_use = P_rigid
            th_use = th_rigid.copy()

        # ---- Phase C: exact LP polish over outer orientation ----
        S_out, beta_out, c_out = _best_orientation(P_use)
        if S_out < best_S and c_out is not None:
            best_S = S_out
            best_result = (P_use, th_use, c_out, beta_out)

    if best_result is None or not _time_ok(1.0):
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
