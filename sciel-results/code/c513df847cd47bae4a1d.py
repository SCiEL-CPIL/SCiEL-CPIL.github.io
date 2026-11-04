"""Heilbronn convex regions: maximize min triangle area / hull area for 13 points.

Strategy: Parametrize 13 points as vertices of a convex polygon on a circle
with fixed uniform angles, optimizing only radial distances. This reduces the
search space from 26D to 13D and guarantees convexity. Then do a final
Cartesian polish to break the circular constraint.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
import itertools

N = 13
MAX_TIME = 145.0  # Safety margin before 150s kill

# Precompute triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_TRI_I = _TRI_IDX[:, 0]
_TRI_J = _TRI_IDX[:, 1]
_TRI_K = _TRI_IDX[:, 2]

# Fixed uniform angles for the circular parametrization
_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS_A = np.cos(_ANGLES)
_SIN_A = np.sin(_ANGLES)


def _hull_area(pts):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _min_tri_area(pts):
    """Calculate the minimum area of any triangle formed by three points."""
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _score(pts):
    """Calculate the score: min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    mta = _min_tri_area(pts)
    return mta / ha


def _radial_to_points(r):
    """Convert radial distances to point coordinates on fixed angles."""
    return np.column_stack([r * _COS_A, r * _SIN_A])


def _radial_objective(r, tau=0.01):
    """
    Objective for radial optimization.
    r: array of N radial distances (all positive)
    Returns negative (min triangle area / hull area) with smooth min.
    """
    pts = _radial_to_points(r)
    
    # Compute hull area (for a star-shaped polygon with sorted angles,
    # the hull is the polygon itself if all r > 0)
    # Use the shoelace formula for the polygon area
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula
    x_shifted = np.roll(x, -1)
    y_shifted = np.roll(y, -1)
    hull_area = 0.5 * np.abs(np.sum(x * y_shifted - x_shifted * y))
    
    if hull_area < 1e-10:
        return 1e6
    
    # Compute all triangle areas
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-12):
        return 1e6
    
    # Smooth minimum
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    smooth_min = max_area - tau * log_sum_exp
    
    score = smooth_min / hull_area
    return -score


def _refine_radial(r0, deadline, max_iter=500):
    """Refine radial distances using L-BFGS-B."""
    best_r = r0.copy()
    best_score = -_radial_objective(r0, tau=0.01)
    
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() > deadline - 3:
            break
        
        def obj(r):
            return _radial_objective(r, tau=temp)
        
        try:
            res = minimize(
                obj,
                best_r,
                method='L-BFGS-B',
                bounds=[(0.1, 2.0)] * N,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_r = res.x
            opt_r = np.clip(opt_r, 0.1, 2.0)
            
            score = -_radial_objective(opt_r, tau=0.0)
            if score > best_score:
                best_score = score
                best_r = opt_r
        except Exception:
            continue
    
    return best_r, best_score


def _cartesian_objective(x, tau=0.005):
    """
    Objective for Cartesian coordinates.
    x: flat array of 2*N coordinates
    Returns negative score (to minimize).
    """
    pts = x.reshape(N, 2)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6
    
    if hull_area < 1e-8:
        return 1e6
    
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas < 1e-12):
        return 1e6
    
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    smooth_min = max_area - tau * log_sum_exp
    
    score = smooth_min / hull_area
    return -score


def _refine_cartesian(pts0, deadline, max_iter=300):
    """Refine in Cartesian space using L-BFGS-B."""
    best_pts = pts0.copy()
    best_score = _score(pts0)
    
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001]
    
    for temp in temperatures:
        if time.time() > deadline - 3:
            break
        
        def obj(x):
            return _cartesian_objective(x, tau=temp)
        
        try:
            res = minimize(
                obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            opt_pts = res.x.reshape(N, 2)
            
            # Check convexity
            try:
                ConvexHull(opt_pts)
            except Exception:
                continue
            
            score = _score(opt_pts)
            if score > best_score:
                best_score = score
                best_pts = opt_pts
        except Exception:
            continue
    
    return best_pts, best_score


def _generate_radial_starts(rng, n_starts=10):
    """Generate diverse radial starting points."""
    starts = []
    
    # Uniform radius (regular polygon)
    starts.append(np.ones(N))
    
    # Slightly perturbed uniform
    for scale in [0.05, 0.1, 0.2, 0.3]:
        r = 1.0 + scale * rng.normal(0, 1, N)
        r = np.clip(r, 0.3, 2.0)
        starts.append(r)
    
    # Alternating radii
    for base in [0.8, 1.0, 1.2]:
        r = np.where(np.arange(N) % 2 == 0, base + 0.3, base - 0.3)
        r = np.clip(r, 0.3, 2.0)
        starts.append(r)
    
    # Random
    for _ in range(n_starts):
        r = 0.5 + 0.8 * rng.random(N)
        starts.append(r)
    
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    best_score = -np.inf
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Radial optimization from multiple starts
    radial_starts = _generate_radial_starts(rng, n_starts=15)
    
    for r0 in radial_starts:
        if time.time() > deadline - 30:
            break
        
        try:
            opt_r, opt_score = _refine_radial(r0, deadline, max_iter=300)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = _radial_to_points(opt_r)
        except Exception:
            continue
    
    # Phase 2: Cartesian polish from best radial solution
    if best_pts is not None and time.time() < deadline - 15:
        try:
            polished_pts, polished_score = _refine_cartesian(best_pts, deadline, max_iter=400)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
        except Exception:
            pass
    
    # Phase 3: Additional Cartesian starts from different radial solutions
    if time.time() < deadline - 10:
        extra_starts = []
        for r0 in radial_starts[:5]:
            r0_clipped = np.clip(r0, 0.3, 2.0)
            extra_starts.append(_radial_to_points(r0_clipped))
        
        for start_pts in extra_starts:
            if time.time() > deadline - 5:
                break
            try:
                polished_pts, polished_score = _refine_cartesian(start_pts, deadline, max_iter=200)
                if polished_score > best_score:
                    best_score = polished_score
                    best_pts = polished_pts
            except Exception:
                continue
    
    # Fallback: regular 13-gon
    if best_pts is None:
        best_pts = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
        best_score = _score(best_pts)
    
    return best_pts
