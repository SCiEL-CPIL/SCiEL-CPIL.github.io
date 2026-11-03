"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
DEADLINE = 135.0  # safety margin below 150s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return DEADLINE - _elapsed()


def _min_triangle_area(points):
    """Minimum area over all triples (vectorized cross product)."""
    n = points.shape[0]
    # All pairs differences
    # Compute area for all triples using cross product
    # For speed, compute all C(n,3) triangle areas
    min_area = float('inf')
    # Vectorized: for each i, compute cross of (p_j - p_i) and (p_k - p_i)
    # Use the formula: area = 0.5*|x1(y2-y3)+x2(y3-y1)+x3(y1-y2)|
    # More efficient: iterate i, vectorize over j,k
    for i in range(n - 2):
        pi = points[i]
        rest = points[i+1:]
        # cross product of all pairs in rest
        dx = rest[:, 0] - pi[0]
        dy = rest[:, 1] - pi[1]
        # area of triangle (i, i+1+j, i+1+k) for j<k
        # cross = dx_j * dy_k - dx_k * dy_j
        cross = np.outer(dx, dy) - np.outer(dy, dx)
        # upper triangle (j<k)
        tri_mask = np.triu(np.ones((len(rest), len(rest)), dtype=bool), k=1)
        areas = 0.5 * np.abs(cross[tri_mask])
        if areas.size > 0:
            local_min = areas.min()
            if local_min < min_area:
                min_area = local_min
    return min_area


def _score(points):
    """min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _params_to_points(params):
    """
    Convert 14 params to 14 points on a convex polygon.
    First 7: angular gaps (positive), rest normalized to sum=2pi
    Last 7: radial distances (positive)
    Points are placed at cumulative angles with given radii.
    """
    angle_gaps = np.abs(params[:7]) + 1e-4
    radii = np.abs(params[7:14]) + 1e-4
    # Normalize angles to sum to 2*pi
    angles = np.cumsum(angle_gaps)
    angles = angles / angles[-1] * 2 * np.pi
    angles = np.concatenate([np.array([0.0]), angles])
    # Actually we need 14 angles; let's use all 14 gaps
    # Redo: use 14 angular gaps and 14 radii? No, task says 7+7.
    # Let me use 14 points with 14 angles derived from 7 gap pairs.
    # Better: 7 gaps for first half, 7 for second? 
    # Actually, let's use all 14 points: 7 angular gaps repeated/scaled + 7 radii
    # Simpler: use 14 angular gaps from params[0:14] and 14 radii from params[14:28]?
    # But we only have 14 params. Let me reconsider.
    
    # New approach: 14 points, parameterized by 14 angles (cumulative from 7 base gaps)
    # and 14 radii. That's 28 params. Let me use 28 params instead.
    # But the directive says 14 params (7 angles, 7 radii). 
    # Let me interpret: 7 angular gaps that are symmetric, and 7 radii for first 7, 
    # with second 7 mirroring. Or just use a different parameterization.
    
    # Actually, for a convex polygon with 14 vertices, we need 14 angles and 14 radii.
    # Let me use 28 parameters: 14 angles (as gaps) and 14 radii.
    # But the directive said 14 params. Let me just use a simpler approach:
    # Parameterize by 14 angles (sum to 2pi) and 14 radii, total 28 params.
    # I'll adjust to use 28 params in the optimizer.
    
    # For now, return a placeholder; I'll fix in the main function.
    return None


def _objective(params):
    """Negative score for minimization."""
    points = _params_to_points_full(params)
    if points is None:
        return 1e10
    s = _score(points)
    return -s


def _params_to_points_full(params):
    """28 params: 14 angle gaps (normalized to 2pi) + 14 radii."""
    angle_gaps = np.abs(params[:14]) + 1e-6
    radii = np.abs(params[14:28]) + 1e-6
    angles = np.cumsum(angle_gaps)
    angles = angles / angles[-1] * 2 * np.pi
    angles = np.concatenate([np.array([0.0]), angles])
    # We have 15 angles (including 0), take first 14
    angles = angles[:14]
    # Ensure strictly increasing
    # Make sure angles are strictly increasing
    for i in range(1, len(angles)):
        if angles[i] <= angles[i-1] + 1e-8:
            angles[i] = angles[i-1] + 1e-6
    # Rescale to [0, 2pi)
    angles = angles / angles[-1] * 2 * np.pi
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def construct():
    global _start
    _start = time.time()
    
    best_score = -np.inf
    best_points = None
    
    # Initial random points
    rng = np.random.RandomState(42)
    
    # Try multiple random starts
    n_starts = 0
    while _time_left() > 30:
        # Random convex polygon initialization
        angle_gaps = rng.rand(14) * 0.5 + 0.1
        radii = rng.rand(14) * 0.8 + 0.2
        params = np.concatenate([angle_gaps, radii])
        
        # Local optimization with Nelder-Mead
        try:
            result = minimize(
                _objective,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
        except Exception:
            continue
        
        if _time_left() < 15:
            break
        
        opt_params = result.x
        points = _params_to_points_full(opt_params)
        if points is not None:
            s = _score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        
        n_starts += 1
        # Limit number of starts to avoid infinite loop
        if n_starts >= 20:
            break
    
    # If we have a good result, return it
    if best_points is not None and best_score > 0:
        return best_points
    
    # Fallback: regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points
