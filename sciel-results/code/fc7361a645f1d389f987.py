"""
Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
- Structural parameterization: represent the 13 points as a regular 13-gon
  distorted by (a) a linear elliptical map (2x2 symmetric matrix) and/or
  (b) per-vertex radial perturbations.  This keeps the configuration close to
  convex and smooth, which is the known regime for Heilbronn configurations.
- Smooth surrogate: log-sum-exp soft-min of triangle areas divided by convex
  hull area, optimized with L-BFGS-B at increasing sharpness (annealing).
- Final polish: Nelder-Mead on the exact (non-smooth) objective.
- Multi-start from diverse structural initializations.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

# Unit circle regular 13-gon
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_BASE_POINTS = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])


def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
        (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    )
    return areas


def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area


def _soft_min_objective(x, beta=80.0):
    """
    Negative soft-min of (min triangle area / hull area) for minimization.
    Uses log-sum-exp for numerical stability.
    x: flattened point coordinates.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    # Soft-min of areas via log-sum-exp
    neg_scaled = -beta * areas
    max_neg = np.max(neg_scaled)
    soft_min_areas = -(max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))) / beta
    return -soft_min_areas / hull_area


def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)


# --- Elliptical distortion parameterization ---

def _elliptic_pts(x):
    """
    Map parameter vector x to 2D points.
    x = [a, b, c, d] where M = [[a, b], [b, c]], d = global scale.
    pts = d * M @ base_points.T  -> (2, N) -> transpose
    """
    a, b, c, d = x
    M = np.array([[a, b], [b, c]])
    pts = d * (M @ _BASE_POINTS.T).T
    return pts


def _elliptic_objective(x, beta=80.0):
    """Soft-min objective for elliptical distortion parameters."""
    pts = _elliptic_pts(x)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    neg_scaled = -beta * areas
    max_neg = np.max(neg_scaled)
    soft_min_areas = -(max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))) / beta
    return -soft_min_areas / hull_area


def _elliptic_exact_objective(x):
    pts = _elliptic_pts(x)
    return -_score(pts)


# --- Radial distortion parameterization ---

def _radial_pts(x):
    """
    Map parameter vector x (length N) to 2D points.
    r_i = 1 + x_i, angles fixed at regular polygon.
    """
    radii = 1.0 + x
    pts = np.column_stack([radii * np.cos(_BASE_ANGLES),
                           radii * np.sin(_BASE_ANGLES)])
    return pts


def _radial_objective(x, beta=80.0):
    """Soft-min objective for radial distortion parameters."""
    pts = _radial_pts(x)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    neg_scaled = -beta * areas
    max_neg = np.max(neg_scaled)
    soft_min_areas = -(max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))) / beta
    return -soft_min_areas / hull_area


def _radial_exact_objective(x):
    pts = _radial_pts(x)
    return -_score(pts)


def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts


def _optimize_with_annealing(objective, x0, betas, max_iter=300, ftol=1e-13, gtol=1e-10):
    """Run L-BFGS-B with increasing beta (annealing). Returns best x."""
    current_x = x0.copy()
    for beta in betas:
        try:
            res = minimize(
                objective,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            current_x = res.x
        except Exception:
            break
    return current_x


def _elliptic_init(seed=0):
    """Initialize elliptic parameters: near-identity with small perturbation."""
    rng = np.random.RandomState(seed)
    # M ≈ I, d ≈ 1
    a = 1.0 + rng.normal(0, 0.1)
    b = rng.normal(0, 0.05)
    c = 1.0 + rng.normal(0, 0.1)
    d = 1.0 + rng.normal(0, 0.05)
    return np.array([a, b, c, d])


def _radial_init(seed=0, noise=0.0):
    """Initialize radial parameters: zero (regular polygon) with optional noise."""
    rng = np.random.RandomState(seed)
    if noise > 0:
        return rng.normal(0, noise, N)
    return np.zeros(N)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    best_score = 0.0
    best_pts = _BASE_POINTS.copy()

    # Evaluate baseline
    base_score = _score(best_pts)
    best_score = base_score

    # Phase 1: Elliptic distortion optimization
    # Multiple starts with different initial perturbations
    n_elliptic_starts = 15
    for i in range(n_elliptic_starts):
        if time.time() > deadline - 40:
            break

        seed = i * 7 + 1
        x0 = _elliptic_init(seed)
        betas = [20, 50, 100, 200, 400]

        x_opt = _optimize_with_annealing(_elliptic_objective, x0, betas,
                                         max_iter=200, ftol=1e-14, gtol=1e-11)
        pts = _elliptic_pts(x_opt)
        score = _score(pts)

        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Phase 2: Radial distortion optimization
    n_radial_starts = 15
    for i in range(n_radial_starts):
        if time.time() > deadline - 30:
            break

        seed = i * 11 + 3
        noise = 0.0 if i < 5 else 0.05 * (i - 4)
        x0 = _radial_init(seed, noise=noise)
        betas = [20, 50, 100, 200, 400]

        x_opt = _optimize_with_annealing(_radial_objective, x0, betas,
                                         max_iter=200, ftol=1e-14, gtol=1e-11)
        pts = _radial_pts(x_opt)
        score = _score(pts)

        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Phase 3: Full 2D Nelder-Mead polish on best solution
    if time.time() < deadline - 20:
        polished = _nelder_mead_polish(best_pts, max_iter=1000, tol=1e-12)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()

    # Phase 4: If time permits, do a second round of elliptic + radial with
    # tighter annealing starting from the current best (converted to params)
    if time.time() < deadline - 10:
        # Try to convert best_pts to elliptic params via least squares
        # pts ≈ d * M @ base_points.T
        # Solve for M and d
        # base_points is (N, 2), pts is (N, 2)
        # pts.T = d * M @ base_points.T
        # Let B = base_points.T (2, N), P = pts.T (2, N)
        # P = d * M @ B  =>  P B^T = d * M @ B B^T
        # If B B^T is invertible: M = P B^T (B B^T)^{-1} / d
        # Try d = 1 first
        B = _BASE_POINTS.T  # (2, N)
        P = best_pts.T      # (2, N)
        BBT = B @ B.T       # (2, 2)
        try:
            M_est = P @ B.T @ np.linalg.inv(BBT)
            a_est, b_est = M_est[0, 0], M_est[0, 1]
            c_est = M_est[1, 1]
            # Estimate d from average radius
            d_est = np.mean(np.linalg.norm(best_pts, axis=1))
            x_ell = np.array([a_est, b_est, c_est, d_est])
            betas = [100, 200, 500]
            x_opt = _optimize_with_annealing(_elliptic_objective, x_ell, betas,
                                             max_iter=150, ftol=1e-15, gtol=1e-12)
            pts = _elliptic_pts(x_opt)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass

    # Final polish
    if time.time() < deadline - 3:
        final_polish = _nelder_mead_polish(best_pts, max_iter=500, tol=1e-12)
        score = _score(final_polish)
        if score > best_score:
            best_score = score
            best_pts = final_polish.copy()

    return best_pts
