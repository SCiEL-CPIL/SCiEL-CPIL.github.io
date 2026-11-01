"""Optimize 14 points in 3D to minimize (max_pairwise_dist / min_pairwise_dist)^2.

Structural change: use EDM (Euclidean Distance Matrix) formulation with
bisection on the ratio, plus smooth objective refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3
_NUM_PAIRS = N * (N - 1) // 2
_DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_sq_dists(pts):
    """Compute all pairwise squared distances (upper triangle, flat)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, 1)
    return sq[iu]


def _raw_score(pts):
    """Compute (max_d / min_d)^2 = max_sq / min_sq."""
    sq = _pairwise_sq_dists(pts)
    min_sq = np.min(sq)
    max_sq = np.max(sq)
    if min_sq < 1e-14:
        return 1e10
    return max_sq / min_sq


def _objective_smooth(x, tau=100.0):
    """Smooth surrogate: 2*(soft_max(log d2) - soft_min(log d2))."""
    pts = x.reshape(N, D)
    sq = _pairwise_sq_dists(pts)
    sq_safe = np.maximum(sq, 1e-15)
    log_sq = np.log(sq_safe)
    
    # Soft max of log_sq
    m = np.max(log_sq)
    soft_max = m + (1.0 / tau) * np.log(np.sum(np.exp(log_sq - m)))
    
    # Soft min of log_sq
    m2 = np.min(log_sq)
    soft_min = m2 - (1.0 / tau) * np.log(np.sum(np.exp(m2 - log_sq)))
    
    return 2.0 * (soft_max - soft_min)


def _edm_penalty(d2, weight=1.0):
    """Penalty for negative eigenvalues of the double-centered Gram matrix.
    d2: flat array of _NUM_PAIRS squared distances.
    Returns penalty value (0 if EDM is valid)."""
    # Reconstruct full distance matrix
    dist_mat = np.zeros((N, N))
    iu = np.triu_indices(N, 1)
    dist_mat[iu] = d2
    dist_mat[iu[1], iu[0]] = d2
    
    # Double-centering: B = -1/2 * J * D2 * J where J = I - (1/n)11^T
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ dist_mat @ J
    
    # Eigenvalues of B; for valid EDM, all should be >= 0 (up to D dimensions)
    eigvals = np.linalg.eigvalsh(B)
    # Penalty: sum of squares of negative eigenvalues
    neg = np.minimum(eigvals, 0.0)
    return weight * np.sum(neg * neg)


def _edm_objective(x, ratio, weight=1.0):
    """Objective for EDM approach: minimize penalty + constraint violation.
    x: flat array of _NUM_PAIRS squared distances.
    ratio: d_max_sq / d_min_sq target.
    We enforce d_min_sq <= d2 <= d_max_sq where d_max_sq = ratio * d_min_sq.
    We fix d_min_sq = 1.0 (scale invariance) so d2 in [1, ratio]."""
    d2 = x
    # Penalty for being outside [1, ratio]
    below = np.maximum(0.0, 1.0 - d2)
    above = np.maximum(0.0, d2 - ratio)
    bound_penalty = np.sum(below * below) + np.sum(above * above)
    
    # EDM feasibility penalty
    edm_pen = _edm_penalty(d2, weight)
    
    return bound_penalty + edm_pen


def _edm_to_points(d2):
    """Convert squared distance matrix to point coordinates via eigendecomposition."""
    dist_mat = np.zeros((N, N))
    iu = np.triu_indices(N, 1)
    dist_mat[iu] = d2
    dist_mat[iu[1], iu[0]] = d2
    
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ dist_mat @ J
    
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep top D eigenvalues (clip negatives)
    k = min(D, N)
    vals = np.maximum(eigvals[:k], 0.0)
    vecs = eigvecs[:, :k]
    
    pts = vecs @ np.diag(np.sqrt(vals))
    # Center
    pts -= np.mean(pts, axis=0)
    return pts


def _make_fibonacci_sphere():
    """Fibonacci sphere initialization."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((N, 3))
    for i in range(N):
        z = 1.0 - (i / (N - 1.0)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i
        pts[i] = [np.cos(theta) * radius, np.sin(theta) * radius, z]
    return pts


def _make_rings():
    """Two rings of 7 points, offset."""
    pts = np.zeros((N, 3))
    z1, r1 = 0.5, np.sqrt(1.0 - 0.25)
    for i in range(7):
        theta = 2 * np.pi * i / 7
        pts[i] = [r1 * np.cos(theta), r1 * np.sin(theta), z1]
    z2, r2 = -0.5, np.sqrt(1.0 - 0.25)
    for i in range(7):
        theta = 2 * np.pi * (i + 0.5) / 7
        pts[i + 7] = [r2 * np.cos(theta), r2 * np.sin(theta), z2]
    return pts


def _make_layers():
    """Three-layer structure: 5-4-5."""
    pts = np.zeros((N, 3))
    z1, r1 = -0.6, np.sqrt(1.0 - 0.36)
    for i in range(5):
        theta = 2 * np.pi * i / 5
        pts[i] = [r1 * np.cos(theta), r1 * np.sin(theta), z1]
    z2, r2 = 0.0, 1.0
    for i in range(4):
        theta = 2 * np.pi * (i + 0.25) / 4
        pts[i + 5] = [r2 * np.cos(theta), r2 * np.sin(theta), z2]
    z3, r3 = 0.6, np.sqrt(1.0 - 0.36)
    for i in range(5):
        theta = 2 * np.pi * (i + 0.2) / 5
        pts[i + 9] = [r3 * np.cos(theta), r3 * np.sin(theta), z3]
    return pts


def _make_stella():
    """Stella octangula (two interlocking tetrahedra) + 6 extra points."""
    v = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    t1 = v.copy()
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t2 = v @ R.T
    pts = np.vstack([t1, t2])
    extra = np.array([
        [0, 0, 2.0], [0, 0, -2.0],
        [2.0, 0, 0], [-2.0, 0, 0],
        [0, 2.0, 0], [0, -2.0, 0]
    ])
    return np.vstack([pts, extra])


def construct():
    start_time = time.time()
    best_pts = None
    best_score = np.inf

    def _update(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # ---- Generate diverse initial configurations ----
    inits = []
    inits.append(_make_fibonacci_sphere())
    inits.append(_make_rings())
    inits.append(_make_layers())
    inits.append(_make_stella())

    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.random((N, 3)) * 4.0 - 2.0
        inits.append(pts)
    for _ in range(2):
        pts = rng.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        inits.append(pts)

    # Evaluate initial configs
    for cfg in inits:
        score = _raw_score(cfg)
        _update(cfg, score)

    if time.time() - start_time > _DEADLINE - 60:
        return best_pts

    # ---- Phase A: EDM-based bisection on ratio ----
    # For each good initial config, try to find a feasible EDM with small ratio
    for cfg in inits[:6]:
        if time.time() - start_time > _DEADLINE - 30:
            break
        pts = cfg - np.mean(cfg, axis=0)
        sq = _pairwise_sq_dists(pts)
        ratio_init = np.max(sq) / np.min(sq) if np.min(sq) > 1e-14 else 10.0

        # Bisection on ratio (in squared distance space)
        lo, hi = 1.0, ratio_init * 1.5
        best_ratio = hi
        best_d2 = None

        for _ in range(20):
            if time.time() - start_time > _DEADLINE - 25:
                break
            mid = (lo + hi) / 2.0
            # Initialize d2: scale sq to [1, mid]
            sq_min = np.min(sq)
            sq_max = np.max(sq)
            if sq_max > sq_min and sq_min > 1e-14:
                d2_init = 1.0 + (sq - sq_min) / (sq_max - sq_min) * (mid - 1.0)
            else:
                d2_init = np.full(_NUM_PAIRS, mid ** 0.5)

            # Optimize d2 to satisfy EDM constraints
            try:
                res = minimize(
                    _edm_objective, d2_init, args=(mid, 10.0),
                    method='L-BFGS-B',
                    bounds=[(1.0, mid)] * _NUM_PAIRS,
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                pen = res.fun
                if pen < 0.01:  # Feasible enough
                    best_ratio = mid
                    best_d2 = res.x
                    hi = mid
                else:
                    lo = mid
            except Exception:
                lo = mid

        if best_d2 is not None:
            pts_edm = _edm_to_points(best_d2)
            score = _raw_score(pts_edm)
            _update(pts_edm, score)

    if time.time() - start_time > _DEADLINE - 20:
        return best_pts

    # ---- Phase B: Smooth objective refinement with tau annealing ----
    tau_schedule = [20.0, 100.0, 500.0, 2000.0]
    starts = [best_pts.flatten()]
    for _ in range(3):
        starts.append(best_pts.flatten() + rng.normal(0, 0.05, size=N * D))

    for x_start in starts:
        if time.time() - start_time > _DEADLINE - 10:
            break
        x_cur = x_start.copy()
        for tau in tau_schedule:
            if time.time() - start_time > _DEADLINE - 8:
                break
            try:
                res = minimize(
                    _objective_smooth, x_cur, args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    x_cur = res.x
            except Exception:
                pass
        pts = x_cur.reshape(N, D)
        score = _raw_score(pts)
        _update(pts, score)

    if time.time() - start_time > _DEADLINE - 5:
        return best_pts

    # ---- Phase C: Nelder-Mead on raw objective for final polish ----
    try:
        res_nm = minimize(
            lambda x: _raw_score(x.reshape(N, D)),
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if res_nm.x is not None:
            pts_nm = res_nm.x.reshape(N, D)
            score_nm = _raw_score(pts_nm)
            _update(pts_nm, score_nm)
    except Exception:
        pass

    # ---- Phase D: High-tau SLSQP polish ----
    if time.time() - start_time < _DEADLINE - 2:
        try:
            res_p = minimize(
                _objective_smooth, best_pts.flatten(), args=(5000.0,),
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if res_p.x is not None:
                pts_p = res_p.x.reshape(N, D)
                score_p = _raw_score(pts_p)
                _update(pts_p, score_p)
        except Exception:
            pass

    if best_pts is None:
        import itertools
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)

    return best_pts
