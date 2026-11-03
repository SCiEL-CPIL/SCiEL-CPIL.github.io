import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 142.0          # safety margin before 150s kill
SQRT3 = math.sqrt(3.0)
INV_SQRT3 = 1.0 / SQRT3
PEN_W = 1000.0


def _hex_vertices_local(theta):
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _all_vertices(centers, theta):
    vl = _hex_vertices_local(theta)
    return (centers[:, None, :] + vl[None, :, :]).reshape(-1, 2)


def _min_enclosing_lp(centers, theta, theta_outer):
    """Exact minimal enclosing regular hexagon (fixed outer angle).
    Returns (cx, cy, apothem) or None."""
    verts = _all_vertices(centers, theta)
    n = verts.shape[0]
    normals = _outer_side_normals(theta_outer)
    A = []
    b = []
    for k in range(6):
        nx, ny = normals[k]
        A.append([-nx, -ny, -1.0])
        b.append(-(nx * verts[:, 0] + ny * verts[:, 1]))
        A.append([nx, ny, -1.0])
        b.append(nx * verts[:, 0] + ny * verts[:, 1])
    A = np.array(A)
    b = np.array(b)
    A = np.vstack([A, np.zeros((n, 3))])
    b = np.concatenate([b, np.zeros(n)])
    try:
        res = linprog([0.0, 0.0, 1.0], A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            return res.x[0], res.x[1], res.x[2]
    except Exception:
        pass
    return None


def _outer_side(theta, centers, cx, cy, theta_outer):
    """Smooth objective for the outer side length (differentiable via argmax)."""
    verts = _all_vertices(centers, theta)
    rel = verts - np.array([cx, cy])
    normals = _outer_side_normals(theta_outer)
    proj = rel @ normals.T          # (n,6)
    apothem = proj.max()
    return 2.0 * apothem * INV_SQRT3


def _pairwise_penalty(centers):
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    viol = np.maximum(0.0, 2.0 - d[iu])
    return np.sum(viol * viol)


def _objective(params):
    """params = [x0,y0,...,xN-1,yN-1, theta, cx, cy, theta_outer]."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    cx = params[N * 2 + 1]
    cy = params[N * 2 + 2]
    theta_outer = params[N * 2 + 3]
    s = _outer_side(theta, centers, cx, cy, theta_outer)
    return s + PEN_W * _pairwise_penalty(centers)


def _objective_grad(params):
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    cx = params[N * 2 + 1]
    cy = params[N * 2 + 2]
    theta_outer = params[N * 2 + 3]

    verts = _all_vertices(centers, theta)
    rel = verts - np.array([cx, cy])
    normals = _outer_side_normals(theta_outer)
    proj = rel @ normals.T
    kmax = int(np.argmax(proj))
    nmax = normals[kmax]
    apothem = proj.max()
    s = 2.0 * apothem * INV_SQRT3

    g = np.zeros_like(params)
    # overlap penalty gradient
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, 1.0)
    viol = np.maximum(0.0, 2.0 - d)
    f = np.zeros_like(d)
    nz = viol > 0
    f[nz] = 2.0 * viol[nz] / d[nz]
    gpen = 2.0 * PEN_W * (f @ diff)                 # (N,2)
    g[:N * 2] = gpen.reshape(-1)

    # smooth outer-side gradient (subgradient at active vertex)
    s2 = 2.0 * INV_SQRT3
    g[N * 2 + 1] = -s2 * nmax[0]
    g[N * 2 + 2] = -s2 * nmax[1]
    g[N * 2 + 3] = -s2 * (rel @ (normals.T @ nmax))

    # d(apothem)/d(theta) = sum_k (nmax . (c_k + R(theta) v_k))
    ang = np.arange(6) * (math.pi / 3.0) + theta
    vl = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    dvl = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    dot = (centers[:, 0] * nmax[0] + centers[:, 1] * nmax[1]) + (vl @ nmax)
    g[N * 2] = s2 * dot.sum()
    g[:N * 2] += (s2 * dot)[:, None] * (nmax[None, :] * dvl).sum(axis=1).reshape(-1)
    return g


def _is_valid(centers, theta, cx, cy, theta_outer, tol=1e-6):
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    verts = _all_vertices(centers, theta)
    rel = verts - np.array([cx, cy])
    normals = _outer_side_normals(theta_outer)
    apothem = (rel @ normals.T).max()
    side = 2.0 * apothem * INV_SQRT3
    return side, True


def _make_init(seed):
    rng = np.random.RandomState(seed)
    mode = seed % 5
    sqrt3 = SQRT3

    if mode == 0:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float)
    elif mode == 3:
        u = (sqrt3, 0.0)
        v = (sqrt3 / 2, 1.5)
        cand = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                cand.append((x * x + y * y, x, y))
        cand.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in cand[:N]])
    else:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float) + rng.uniform(-0.1, 0.1, size=(N, 2))

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    cx, cy = 0.0, 0.0
    theta_outer = rng.uniform(0.0, math.pi / 6.0)
    return np.concatenate([centers.reshape(-1), [theta, cx, cy, theta_outer]])


def _optimize_one(init_params, maxiter):
    try:
        res = minimize(_objective, init_params, jac=_objective_grad,
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
        return res
    except Exception:
        return None


def _polish(params):
    """High-precision polish on the smooth objective."""
    try:
        res = minimize(_objective, params, jac=_objective_grad,
                       method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12})
        if (res.success or res.fun < _objective(params)):
            return res.x
    except Exception:
        pass
    return params


def _finalize(centers, theta, cx, cy, theta_outer):
    """Exact LP verification; returns (cx, cy, side) or None."""
    r = _min_enclosing_lp(centers, theta, theta_outer)
    if r is None:
        return None
    cx2, cy2, apothem = r
    side = 2.0 * apothem * INV_SQRT3
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return None
    return cx2, cy2, side


def construct():
    best = None
    best_score = float('inf')
    best_params = None

    # diverse outer-angle starting points to escape local minima
    base_rot = [0.0, math.pi / 6.0, math.pi / 12.0, math.pi / 3.0, math.pi / 4.0]

    seed = 0
    maxiter = 300
    while time.time() < DEADLINE - 12:
        init = _make_init(seed)
        # bias the outer angle toward a diverse base rotation
        init[N * 2 + 3] = base_rot[seed % len(base_rot)] + (init[N * 2 + 3] % (math.pi / 6.0)) * 0.0
        seed += 1
        res = _optimize_one(init, maxiter)
        if res is None:
            continue
        p = res.x
        centers = p[:N * 2].reshape(N, 2)
        theta = p[N * 2]
        cx, cy, theta_o = p[N * 2 + 1], p[N * 2 + 2], p[N * 2 + 3]
        fin = _finalize(centers, theta, cx, cy, theta_o)
        if fin is None:
            continue
        cx, cy, s = fin
        if s < best_score:
            best_score = s
            best = (centers.copy(), theta, cx, cy, theta_o)
            best_params = p.copy()
        if time.time() > DEADLINE - 18:
            break

    # polish best
    if best_params is not None and time.time() < DEADLINE - 2:
        p = _polish(best_params)
        centers = p[:N * 2].reshape(N, 2)
        theta = p[N * 2]
        cx, cy, theta_o = p[N * 2 + 1], p[N * 2 + 2], p[N * 2 + 3]
        fin = _finalize(centers, theta, cx, cy, theta_o)
        if fin is not None:
            cx, cy, s = fin
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, cx, cy, theta_o)

    if best is None:
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, cx, cy, theta_o = best
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (float(cx), float(cy)), float(best_score), float(math.degrees(theta_o))
