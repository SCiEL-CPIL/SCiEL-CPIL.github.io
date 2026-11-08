"""Heilbronn 13-point problem: symmetry-constrained two-shell design + DE + local polish."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE_SAFETY = 10.0  # return at least this many seconds before the 150s limit


def _min_triangle_area(pts):
    """Exact minimum triangle area over all C(13,3)=286 triples."""
    n = pts.shape[0]
    min_a = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                a = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if a < min_a:
                    min_a = a
    return min_a


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        o = pts[np.argsort(ang)]
        n = len(o)
        s = 0.0
        for i in range(n):
            j = (i + 1) % n
            s += o[i][0] * o[j][1] - o[j][0] * o[i][1]
        return abs(s) * 0.5


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


def _soft_min_area(pts, T=0.003):
    """Smooth approximation of min triangle area via log-sum-exp."""
    n = pts.shape[0]
    areas = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                areas[i, k] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    # gather only upper-triangle entries
    idx = np.triu_indices(n, k=2)
    vals = areas[idx]
    s = -vals / T
    m = s.max()
    return -(m + np.log(np.sum(np.exp(s - m)))) * T


def _build_two_shell(params):
    """
    Build 13 points from 7 parameters:
      p[0] = inner radius r1
      p[1] = outer radius r2
      p[2] = inner angular offset (radians)
      p[3] = outer angular offset (radians)
      p[4] = inner radius modulation amplitude (small)
      p[5] = outer radius modulation amplitude (small)
      p[6] = modulation frequency (integer-ish, use 1 or 2)
    Layout: center (0,0); 6 inner at angles i*pi/3 + offset1;
            6 outer at angles i*pi/3 + offset2 + pi/6 (alternating).
    """
    r1, r2, a1, a2, mod1, mod2, freq = params
    pts = np.zeros((13, 2))
    # center
    pts[0] = [0.0, 0.0]
    # inner ring: 6 points
    for i in range(6):
        ang = i * np.pi / 3.0 + a1
        r = r1 + mod1 * np.cos(freq * ang)
        pts[1 + i] = [r * np.cos(ang), r * np.sin(ang)]
    # outer ring: 6 points, offset by pi/6 to interleave
    for i in range(6):
        ang = i * np.pi / 3.0 + a2 + np.pi / 6.0
        r = r2 + mod2 * np.cos(freq * ang)
        pts[7 + i] = [r * np.cos(ang), r * np.sin(ang)]
    return pts


def _obj_constrained(params):
    """Objective for DE: minimize negative score of two-shell config."""
    pts = _build_two_shell(params)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    min_a = _min_triangle_area(pts)
    return -min_a / ha


def _obj_full(x):
    """Objective for local polish on all 26 coordinates."""
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    return -_min_triangle_area(pts) / ha


def _obj_smooth_full(x, T=0.002):
    """Smooth objective for L-BFGS-B."""
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 1e10
    return -_soft_min_area(pts, T) / ha


def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_SAFETY

    rng = np.random.default_rng(12345)

    # --- Phase 1: Differential Evolution on 7-parameter two-shell family ---
    # Bounds: [r1, r2, a1, a2, mod1, mod2, freq]
    bounds = [
        (0.15, 0.65),   # r1: inner radius
        (0.70, 1.30),   # r2: outer radius
        (-np.pi / 3, np.pi / 3),  # a1
        (-np.pi / 3, np.pi / 3),  # a2
        (0.0, 0.08),    # mod1
        (0.0, 0.08),    # mod2
        (1.0, 2.0),     # freq (effectively 1 or 2)
    ]

    best_params = None
    best_score = -np.inf
    best_pts = None

    # DE with modest population to fit in time
    try:
        de_res = differential_evolution(
            _obj_constrained,
            bounds=bounds,
            seed=rng,
            maxiter=60,
            popsize=15,
            tol=1e-8,
            polish=False,  # we do our own polish
            init='latinhypercube',
            updating='immediate',
            workers=1,
        )
        if time.time() < deadline:
            best_params = de_res.x
            best_pts = _build_two_shell(best_params)
            best_score = _score(best_pts)
    except Exception:
        pass

    # Fallback / additional: try a few hand-tuned symmetric configs
    fallback_configs = [
        # Regular 13-gon
        np.column_stack((np.cos(np.linspace(0, 2*np.pi, 13, endpoint=False)),
                         np.sin(np.linspace(0, 2*np.pi, 13, endpoint=False)))),
        # Center + 12-gon
        np.vstack([[0, 0],
                   np.column_stack((np.cos(np.linspace(0, 2*np.pi, 12, endpoint=False)),
                                    np.sin(np.linspace(0, 2*np.pi, 12, endpoint=False)))*1.0)]),
        # Center + hexagon(r=0.5) + hexagon(r=1.0, offset 30deg)
        np.vstack([[0, 0],
                   np.column_stack((np.cos(np.linspace(0, 2*np.pi, 6, endpoint=False))*0.5,
                                    np.sin(np.linspace(0, 2*np.pi, 6, endpoint=False))*0.5)),
                   np.column_stack((np.cos(np.linspace(np.pi/6, 2*np.pi+np.pi/6, 6, endpoint=False))*1.0,
                                    np.sin(np.linspace(np.pi/6, 2*np.pi+np.pi/6, 6, endpoint=False))*1.0))]),
    ]
    for fc in fallback_configs:
        if time.time() > deadline:
            break
        s = _score(fc)
        if s > best_score:
            best_score = s
            best_pts = fc.copy()

    if best_pts is None:
        best_pts = fallback_configs[0].copy()
        best_score = _score(best_pts)

    # --- Phase 2: Local polish with L-BFGS-B on full 26 coordinates ---
    if time.time() < deadline:
        x0 = best_pts.flatten()
        try:
            res = minimize(
                _obj_smooth_full, x0, args=(0.002,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10},
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass

    # --- Phase 3: Nelder-Mead fine polish on full coordinates ---
    if time.time() < deadline:
        try:
            res = minimize(
                _obj_full, best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass

    # --- Phase 4: One more smooth L-BFGS-B pass with lower temperature ---
    if time.time() < deadline:
        try:
            res = minimize(
                _obj_smooth_full, best_pts.flatten(), args=(0.0005,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11},
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass

    # Final safety: ensure 13 points, 2D
    best_pts = np.asarray(best_pts, dtype=float).reshape(13, 2)
    return best_pts
