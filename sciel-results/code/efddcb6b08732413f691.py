import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _triangle_areas_vectorized(points):
    """Calculate areas of all C(n,3) triangles vectorized."""
    n = len(points)
    idx = np.array(list(combinations(range(n), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _min_triangle_area(points):
    """Find the minimum area among all triangles formed by the points."""
    return np.min(_triangle_areas_vectorized(points))

def _hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _ratio(points):
    """Calculate the score: min triangle area / hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, temperature=0.05):
    """Soft-min approximation of minimum triangle area (differentiable)."""
    areas = _triangle_areas_vectorized(points)
    areas_clipped = np.clip(areas, 1e-10, None)
    log_sum_exp = -temperature * np.log(np.sum(np.exp(-areas_clipped / temperature)))
    return log_sum_exp

def _soft_ratio(flat_points, temperature=0.05):
    """Soft ratio for optimization."""
    points = flat_points.reshape(-1, 2)
    soft_min_area = _soft_min_triangle_area(points, temperature)
    hull_area = _hull_area(points)
    if hull_area < 1e-15:
        return 0.0
    return soft_min_area / hull_area

def _neg_soft_ratio(flat_points, temperature=0.05):
    """Negative soft ratio for minimization."""
    return -_soft_ratio(flat_points, temperature)

def _neg_ratio(flat_points):
    """Negative ratio for minimization."""
    points = flat_points.reshape(-1, 2)
    return -_ratio(points)

def _regular_polygon(n, radius=1.0, center=(0, 0)):
    """Create a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([center[0] + radius * np.cos(angles),
                           center[1] + radius * np.sin(angles)])
    return pts

def _concentric_rings(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Create points on two concentric rings."""
    outer = _regular_polygon(n_outer, r_outer)
    inner = _regular_polygon(n_inner, r_inner)
    return np.vstack([outer, inner])

def _optimize_from_init(init_points, start_time, time_limit, max_iter_soft=300, max_iter_nm=200):
    """Multi-phase optimization from a given initialization."""
    x0 = init_points.flatten()
    best_points = init_points.copy()
    best_score = _ratio(init_points)
    
    # Phase 1: Soft-min with L-BFGS-B (smooth, gradient-based)
    for temp in [0.1, 0.05, 0.02]:
        if time.time() - start_time > time_limit - 60:
            break
        try:
            result = minimize(
                _neg_soft_ratio,
                x0,
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': max_iter_soft, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < -best_score * 0.99:
                current_points = result.x.reshape(-1, 2)
                current_score = _ratio(current_points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    x0 = current_points.flatten()
        except Exception:
            continue
    
    # Phase 2: Nelder-Mead polishing on the raw (non-smooth) objective
    if time.time() - start_time < time_limit - 30:
        try:
            result = minimize(
                _neg_ratio,
                x0,
                method='Nelder-Mead',
                options={'maxiter': max_iter_nm, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            current_points = result.x.reshape(-1, 2)
            current_score = _ratio(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        except Exception:
            pass
    
    # Phase 3: Final soft-min polish with very low temperature
    if time.time() - start_time < time_limit - 15:
        try:
            result = minimize(
                _neg_soft_ratio,
                best_points.flatten(),
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            current_points = result.x.reshape(-1, 2)
            current_score = _ratio(current_points)
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        except Exception:
            pass
    
    return best_points, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 580  # Safety margin before 630s kill
    
    best_points = None
    best_score = -float('inf')
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13))
    
    # 2. 7 outer + 6 inner (concentric rings)
    initializations.append(_concentric_rings(7, 6, r_outer=1.0, r_inner=0.5))
    
    # 3. 6 outer + 7 inner
    initializations.append(_concentric_rings(6, 7, r_outer=1.0, r_inner=0.4))
    
    # 4. 5 outer + 8 inner
    initializations.append(_concentric_rings(5, 8, r_outer=1.0, r_inner=0.35))
    
    # 5. Regular 12-gon + center
    outer12 = _regular_polygon(12)
    center = np.array([[0.0, 0.0]])
    initializations.append(np.vstack([outer12, center]))
    
    # 6. Perturbed regular 13-gon
    rng = np.random.RandomState(42)
    perturbed = _regular_polygon(13) + rng.normal(0, 0.05, (13, 2))
    initializations.append(perturbed)
    
    # 7. 7 outer + 5 inner + 1 center
    outer7 = _regular_polygon(7)
    inner5 = 0.45 * _regular_polygon(5)
    center = np.array([[0.0, 0.0]])
    initializations.append(np.vstack([outer7, inner5, center]))
    
    for i, init in enumerate(initializations):
        if time.time() - start_time > time_limit - 30:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 30:
            break
        
        try:
            points, score = _optimize_from_init(
                init, start_time, time_limit,
                max_iter_soft=300,
                max_iter_nm=200
            )
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            continue
    
    # Fallback
    if best_points is None:
        best_points = _regular_polygon(13)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13)
    
    return best_points

# EVOLVE-BLOCK-END
