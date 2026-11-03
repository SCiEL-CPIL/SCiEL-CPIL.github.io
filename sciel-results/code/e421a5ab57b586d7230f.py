import time
import math
import numpy as np

T0 = time.time()
DEADLINE = T0 + 140.0          # safety margin before the 150 s kill
SQRT3 = math.sqrt(3.0)
PI = math.pi


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta. (6,2)."""
    ang = np.arange(6) * (PI / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of a regular hexagon rotated by theta_outer. (6,2)."""
    ang = theta_outer + np.arange(6) * (PI / 3.0) + PI / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _compute_side_lp(centers, theta, theta_outer):
    """
    Minimum side length of a regular hexagon (free center + fixed angle theta_outer)
    that encloses all vertices of the inner hexagons.
    LP variables: (cx, cy, a). Constraint: n_i . (v_j - c) <= a  for all (i,j).
    Objective: minimize a.  side = 2 a / sqrt(3).
    """
    from scipy.optimize import linprog
    verts_local = _hex_vertices_local(theta)                      # (6,2)
    verts = centers[:, None, :] + verts_local[None, :, :]         # (N,6,2)
    verts = verts.reshape(-1, 2)                                  # (M,2)
    normals = _outer_side_normals(theta_outer)                    # (6,2)
    M = verts.shape[0]

    A_ub = []
    b_ub = []
    for i in range(6):
        n = normals[i]                                            # (2,)
        row = np.zeros((M, 3))
        row[:, 0] = n[0]
        row[:, 1] = n[1]
        row[:, 2] = -1.0
        A_ub.append(row)
        b_ub.append((n @ verts.T))                                # (M,)
    A_ub = np.vstack(A_ub)
    b_ub = np.concatenate(b_ub)

    c = np.array([0.0, 0.0, 1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0.0, None)])
    if not res.success:
        # Fallback: centered hexagon side length
        proj = verts @ normals.T
        apothem = proj.max()
        return apothem * 2.0 / SQRT3
    a = float(res.x[2])
    return a * 2.0 / SQRT3


def _pairwise_penalty(centers, weight=1000.0):
    """Penalty for pairs closer than 2 (minimum center distance for non-overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return weight * np.sum(viol * viol)


def _objective(params):
    """
    Minimize outer side length (LP) + overlap penalty.
    params: [x0,y0,...,x_{N-1},y_{N-1}, theta_inner, theta_outer]
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    theta_outer = params[N * 2 + 1]
    s = _compute_side_lp(centers, theta, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment (LP side must cover all vertices)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    s = _compute_side_lp(centers, theta, theta_outer)
    apothem = s * SQRT3 / 2.0
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    # Re-solve to get the enclosing center for a true containment check
    from scipy.optimize import linprog
    M = verts.shape[0]
    A_ub = []
    b_ub = []
    for i in range(6):
        n = normals[i]
        row = np.zeros((M, 3))
        row[:, 0] = n[0]
        row[:, 1] = n[1]
        row[:, 2] = -1.0
        A_ub.append(row)
        b_ub.append(n @ verts.T)
    A_ub = np.vstack(A_ub)
    b_ub = np.concatenate(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0.0, None)])
    if not res.success:
        return False
    a = float(res.x[2])
    if a > apothem + tol:
        return False
    return True


def _make_init(seed):
    """Generate diverse initial configurations for the inner hexagons."""
    rng = np.random.RandomState(seed)
    mode = seed % 6

    if mode == 0:
        # Hexagonal lattice: center + ring of 6 + 4 more
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        # Offset hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0 + PI / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        # Compact cluster (known good packing)
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
    elif mode == 3:
        # Slightly perturbed compact cluster
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = [positions[i][0] + rng.normal(0, 0.1),
                          positions[i][1] + rng.normal(0, 0.1)]
    elif mode == 4:
        # Ring-based: 1 center + 6 inner ring + 4 outer ring
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            r = 2.0
            centers[i + 1] = [r * math.cos(ang), r * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            r = 3.2
            centers[i + 7] = [r * math.cos(ang), r * math.sin(ang)]
    else:
        # Random with moderate spread
        centers = rng.uniform(-3, 3, size=(N, 2))

    theta = rng.uniform(-PI / 6, PI / 6)
    theta_outer = rng.uniform(-PI / 6, PI / 6)
    params = np.concatenate([centers.reshape(-1), [theta, theta_outer]])
    return params


def _optimize_one(init_params, maxiter=300):
    """Run L-BFGS-B optimization on (centers, theta_inner, theta_outer)."""
    from scipy.optimize import minimize
    try:
        res = minimize(_objective, init_params,
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x
    except Exception:
        return init_params


def _polish(params):
    """High-precision polish: Nelder-Mead on the nonsmooth objective + tight L-BFGS-B."""
    from scipy.optimize import minimize
    best = params
    best_val = _objective(params)

    # Nelder-Mead handles the piecewise-linear LP objective well
    try:
        res = minimize(_objective, best, method='Nelder-Mead',
                       options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    # Tight L-BFGS-B
    try:
        res = minimize(_objective, best, method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    return best


def construct():
    global N
    N = 11
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    best_params = None

    # A few outer rotation seeds; the LP also optimizes center, so fewer angles suffice
    outer_rotations = [0.0, PI / 6.0, PI / 3.0, PI / 12.0]

    seed = 0
    maxiter = 300

    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 25:   # reserve 25 s for polish
            init_params = _make_init(seed)
            seed += 1
            try:
                params = _optimize_one(init_params, maxiter)
            except Exception:
                continue

            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            theta_outer = params[N * 2 + 1]

            if not _is_valid(centers, theta, theta_outer):
                continue

            s = _compute_side_lp(centers, theta, theta_outer)
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, theta_outer)
                best_params = params.copy()

            if time.time() > DEADLINE - 30:
                break

    # Polish the best solution
    if best_params is not None and time.time() < DEADLINE:
        polished = _polish(best_params)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        theta_o_p = polished[N * 2 + 1]

        if _is_valid(centers_p, theta_p, theta_o_p):
            s_p = _compute_side_lp(centers_p, theta_p, theta_o_p)
            if s_p < best_score:
                best_score = s_p
                best = (centers_p.copy(), theta_p, theta_o_p)

    if best is None:
        # Fallback: return a valid (but suboptimal) configuration
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
        s = _compute_side_lp(centers, 0.0, 0.0)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(s), 0.0

    centers, theta, theta_o = best
    s = _compute_side_lp(centers, theta, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))


N = 11
