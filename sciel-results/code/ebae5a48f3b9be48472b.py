"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _triangle_area(p1, p2, p3):
    """Area of triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area_over_hull(points):
    """
    Compute min triangle area / convex hull area for given points.
    Returns the ratio (higher is better).
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Find minimum triangle area over all triples
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _neg_objective(flat_params):
    """Objective for minimizer: negative of min_triangle_area_over_hull."""
    points = flat_params.reshape(-1, 2)
    return -_min_triangle_area_over_hull(points)


def construct():
    # Start with regular 14-gon centered at origin, circumradius 1
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # Use Nelder-Mead to maximize min triangle area / hull area
    result = minimize(
        _neg_objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    best_points = result.x.reshape(-1, 2)
    
    # Also try a few other initializations and take the best
    candidates = [best_points]
    
    # Try regular polygon with slight rotation
    for rot in [0.1, 0.3, 0.5]:
        angles_rot = 2 * np.pi * np.arange(N) / N + rot
        x0_rot = np.column_stack([np.cos(angles_rot), np.sin(angles_rot)]).flatten()
        res_rot = minimize(
            _neg_objective,
            x0_rot,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        candidates.append(res_rot.x.reshape(-1, 2))
    
    # Try a slightly perturbed regular polygon
    rng = np.random.RandomState(42)
    x0_pert = x0 + 0.05 * rng.randn(N * 2)
    res_pert = minimize(
        _neg_objective,
        x0_pert,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    candidates.append(res_pert.x.reshape(-1, 2))
    
    # Pick the best candidate
    best_score = -1
    best_points = candidates[0]
    for cand in candidates:
        score = _min_triangle_area_over_hull(cand)
        if score > best_score:
            best_score = score
            best_points = cand
    
    return best_points
# EVOLVE-BLOCK-END
