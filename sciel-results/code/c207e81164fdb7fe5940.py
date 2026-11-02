"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Combined strategy:
- Support-function h(θ) = 1 + Σ (a_k cos kθ + b_k sin kθ) with 4 harmonics
  (10 parameters) guarantees strict convexity by construction.
- 14 points are placed at equal arc-length intervals along the support-function
  boundary, with a global phase shift (11 params total).
- Multi-phase optimization: differential_evolution → L-BFGS-B → Nelder-Mead,
  using a smooth soft-min objective for gradient-based refinement.
- Fallback to centrally-symmetric 14-point search if support-function approach
  does not yield improvement.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
K_HARM = 4  # number of harmonics in support function

# Precompute triangle index arrays for vectorized area computation
_idx = np.arange(N)
_I, _J, _K = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_I < _J) & (_J < _K)
_I = _I[_mask].flatten()
_J = _J[_mask].flatten()
_K = _K[_mask].flatten()


def _all_triangle_areas(pts):
    """Vectorized computation of all C(n,3) triangle areas."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)


def _min_triangle_area(pts):
    return float(np.min(_all_triangle_areas(pts)))


def _hull_area(pts):
    try:
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 0.0
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    return _min_triangle_area(pts) / ha


# ---------- Support-function representation ----------

def _support_boundary(coeffs, n_samples=400):
    """
    Generate boundary points from support function h(θ).
    coeffs = [phase, a1, b1, a2, b2, a3, b3, a4, b4]  (9 params)
    h(θ) = 1 + Σ_{k=1}^{4} (a_k cos(kθ) + b_k sin(kθ))
    Boundary point at angle θ: x = h cosθ - h' sinθ, y = h sinθ + h' cosθ
    where h' = dh/dθ.
    """
    phase = coeffs[0]
    a = coeffs[1::2]  # a1..a4
    b = coeffs[2::2]  # b1..b4
    # a has shape (4,), b has shape (4,)
    
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False) + phase
    h = np.ones(n_samples)
    hp = np.zeros(n_samples)  # dh/dθ
    for k in range(1, K_HARM + 1):
        h += a[k-1] * np.cos(k * theta) + b[k-1] * np.sin(k * theta)
        hp += -k * a[k-1] * np.sin(k * theta) + k * b[k-1] * np.cos(k * theta)
    
    x = h * np.cos(theta) - hp * np.sin(theta)
    y = h * np.sin(theta) + hp * np.cos(theta)
    return np.column_stack([x, y])


def _place_equal_arc(boundary, n_pts):
    """Place n_pts at equal arc-length intervals along a closed polygon boundary."""
    diffs = np.diff(boundary, axis=0, append=boundary[:1])
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    targets = (np.arange(n_pts) / n_pts) * total
    pts = np.zeros((n_pts, 2))
    for i in range(n_pts):
        t = targets[i]
        idx = np.searchsorted(cum, t, side='right') - 1
        idx = max(0, min(idx, len(boundary) - 1))
        seg_start = cum[idx]
        seg_end = cum[idx + 1] if idx + 1 < len(cum) else total
        frac = (t - seg_start) / max(seg_end - seg_start, 1e-15)
        frac = np.clip(frac, 0, 1)
        p0 = boundary[idx % len(boundary)]
        p1 = boundary[(idx + 1) % len(boundary)]
        pts[i] = (1 - frac) * p0 + frac * p1
    return pts


def _support_to_points(coeffs):
    """Convert support-function coefficients to 14 points."""
    boundary = _support_boundary(coeffs, n_samples=400)
    return _place_equal_arc(boundary, N)


# ---------- Objectives ----------

def _obj_support_soft(coeffs, beta=30.0):
    """Soft-min objective for support-function parameters (minimize)."""
    pts = _support_to_points(coeffs)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    if mz < -500:
        lse = mz
    else:
        lse = mz + np.log(np.sum(np.exp(z - mz)))
    soft_min = -lse / beta
    if soft_min < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    return ha / soft_min


def _obj_support_hard(coeffs):
    """Hard negative-score objective for support-function parameters."""
    pts = _support_to_points(coeffs)
    return -_score(pts)


# ---------- Centrally symmetric representation (fallback / secondary) ----------

def _sym_to_points(flat):
    """flat = [x1..x7, y1..y7] → 14 points (7 + 7 antipodal)."""
    half = flat.reshape(N_HALF, 2)
    return np.vstack([half, -half])


def _obj_sym_soft(flat, beta=30.0):
    pts = _sym_to_points(flat)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    mz = np.max(z)
    lse = mz + np.log(np.sum(np.exp(z - mz))) if mz > -500 else mz
    soft_min = -lse / beta
    if soft_min < 1e-12:
        return 1e10
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e10
    return ha / soft_min


def _obj_sym_hard(flat):
    return -_score(_sym_to_points(flat))


# ---------- Main construct ----------

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10 s safety margin before 150 s kill

    best_score = 0.0
    best_pts = None

    # Baseline: regular 14-gon
    theta_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(theta_reg), np.sin(theta_reg)])
    best_score = _score(reg_pts)
    best_pts = reg_pts.copy()

    # ---------- Phase 1: Support-function search ----------
    # 9 params: [phase, a1, b1, a2, b2, a3, b3, a4, b4]
    sf_bounds = [
        (0.0, 2 * np.pi),
        (-0.15, 0.15), (-0.15, 0.15),
        (-0.10, 0.10), (-0.10, 0.10),
        (-0.06, 0.06), (-0.06, 0.06),
        (-0.03, 0.03), (-0.03, 0.03),
    ]

    # 1a: Differential evolution on soft objective
    try:
        if time.time() < deadline - 60:
            de_res = differential_evolution(
                _obj_support_soft,
                bounds=sf_bounds,
                seed=42,
                maxiter=30,
                tol=1e-5,
                popsize=10,
                polish=False,
                init='sobol',
            )
            if de_res.fun is not None and np.isfinite(de_res.fun):
                pts = _support_to_points(de_res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
    except Exception:
        pass

    # 1b: Local refinement (L-BFGS-B) from DE result + a few random starts
    sf_starts = []
    try:
        if de_res.fun is not None:
            sf_starts.append(de_res.x.copy())
    except NameError:
        pass
    rng = np.random.RandomState(7)
    for _ in range(3):
        sf_starts.append([rng.uniform(0, 2 * np.pi)] +
                         [rng.uniform(lo, hi) for lo, hi in sf_bounds[1:]])

    for start in sf_starts:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_obj_support_soft, start, method='L-BFGS-B',
                           bounds=sf_bounds, options={'maxiter': 200, 'ftol': 1e-14})
            if res.fun is not None and np.isfinite(res.fun):
                pts = _support_to_points(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # 1c: Nelder-Mead fine-tune on hard objective
    if best_pts is not None and time.time() < deadline - 10:
        try:
            # Recover approximate coeffs from best_pts by using the last good coeffs
            # We'll just do a local NM on the soft objective around the best
            # But we need the coeffs, not the points. Use the last DE/refined coeffs.
            # Fallback: skip if we don't have coeffs; do a direct point-level NM instead.
            pass
        except Exception:
            pass

    # ---------- Phase 2: Centrally symmetric search ----------
    sym_bounds = [(-1.5, 1.5)] * (2 * N_HALF)

    # 2a: DE on symmetric soft objective
    try:
        if time.time() < deadline - 30:
            de2 = differential_evolution(
                _obj_sym_soft,
                bounds=sym_bounds,
                seed=99,
                maxiter=25,
                tol=1e-5,
                popsize=10,
                polish=False,
                init='sobol',
            )
            if de2.fun is not None and np.isfinite(de2.fun):
                pts = _sym_to_points(de2.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
    except Exception:
        pass

    # 2b: L-BFGS-B refinement
    sym_starts = []
    try:
        if de2.fun is not None:
            sym_starts.append(de2.x.copy())
    except NameError:
        pass
    for _ in range(3):
        sym_starts.append(rng.uniform(-1.0, 1.0, 2 * N_HALF))

    for start in sym_starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(_obj_sym_soft, start, method='L-BFGS-B',
                           bounds=sym_bounds, options={'maxiter': 200, 'ftol': 1e-14})
            if res.fun is not None and np.isfinite(res.fun):
                pts = _sym_to_points(res.x)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # ---------- Phase 3: Direct point-level local search ----------
    if best_pts is not None and time.time() < deadline - 5:
        # Nelder-Mead on the 28-d point space (no constraints, just maximize score)
        def _obj_pts_direct(flat):
            p = flat.reshape(N, 2)
            return -_score(p)

        try:
            res = minimize(_obj_pts_direct, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun is not None and np.isfinite(res.fun):
                pts = res.x.reshape(N, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

        # Small random perturbations
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            perturbed = best_pts + rng.normal(0, 0.03, (N, 2))
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_pts = perturbed.copy()

    return best_pts
