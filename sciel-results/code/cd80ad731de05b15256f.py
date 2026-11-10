# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.optimize import minimize, differential_evolution
import itertools

_DEADLINE = time.time() + 625.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _smooth_min_area_objective(pts_flat, alpha=50.0):
    """
    Objective function for optimization.
    Returns negative of the smoothed minimum area divided by hull area.
    We want to maximize A_min / A_hull, so we minimize - (A_min / A_hull).
    Using log-sum-exp to approximate min(areas) smoothly.
    """
    pts = pts_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        
        n = len(pts)
        idx = np.array(list(itertools.combinations(range(n), 3)))
        a = pts[idx[:, 0]]
        b = pts[idx[:, 1]]
        c = pts[idx[:, 2]]
        
        # Calculate all triangle areas
        areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                             + b[:, 0] * (c[:, 1] - a[:, 1])
                             + c[:, 0] * (a[:, 1] - b[:, 1]))
        
        # Check for degenerate triangles
        if np.any(areas < 1e-10):
            return 1e6

        # Smooth minimum: -1/alpha * log(sum(exp(-alpha * areas)))
        # To avoid overflow, subtract max
        max_neg = np.max(-alpha * areas)
        log_sum_exp = np.log(np.sum(np.exp(-alpha * areas - max_neg))) + max_neg
        smooth_min = -log_sum_exp / alpha
        
        # We want to maximize smooth_min / ha
        # So minimize - smooth_min / ha
        return -(smooth_min / ha)
    except Exception:
        return 1e6

def _nelder_mead_smooth_optimize(points0, max_iter=2000, alpha=50.0):
    """Nelder-Mead optimization using smooth min area objective."""
    def objective(pts_flat):
        return _smooth_min_area_objective(pts_flat, alpha)
    
    try:
        result = minimize(
            objective,
            points0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return points0

def _jittered_regular_polygon(n, radius, center, jitter_scale):
    """Generate a regular polygon with random jitter to break symmetry."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Add random jitter to angles and radii
    angles += np.random.uniform(-jitter_scale, jitter_scale, n)
    radii = radius * (1 + np.random.uniform(-jitter_scale, jitter_scale, n))
    
    pts = np.column_stack([center[0] + radii * np.cos(angles),
                           center[1] + radii * np.sin(angles)])
    return pts

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Use multiple starting configurations
    # 1. Regular 12-gon + center
    # 2. Regular 13-gon
    # 3. Jittered Regular 12-gon + center
    # 4. Jittered Regular 13-gon
    
    starting_configs = []
    
    # Config 1: Regular 12-gon + center
    n_hull = 12
    angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    radius = 0.45
    center = (0.5, 0.5)
    hull_pts = np.column_stack([center[0] + radius * np.cos(angles),
                                center[1] + radius * np.sin(angles)])
    start1 = np.vstack([hull_pts, np.array([center])])
    starting_configs.append(start1)
    
    # Config 2: Regular 13-gon
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    radius13 = 0.45
    start2 = np.column_stack([center[0] + radius13 * np.cos(angles13),
                              center[1] + radius13 * np.sin(angles13)])
    starting_configs.append(start2)
    
    # Config 3: Jittered Regular 12-gon + center
    np.random.seed(42)
    start3 = _jittered_regular_polygon(12, 0.45, (0.5, 0.5), 0.1)
    start3 = np.vstack([start3, np.array([0.5, 0.5])])
    starting_configs.append(start3)
    
    # Config 4: Jittered Regular 13-gon
    start4 = _jittered_regular_polygon(13, 0.45, (0.5, 0.5), 0.1)
    starting_configs.append(start4)
    
    # Config 5: Random points in unit square (clipped)
    np.random.seed(123)
    start5 = np.random.uniform(0.1, 0.9, (13, 2))
    starting_configs.append(start5)

    for start_pts in starting_configs:
        if time.time() > _DEADLINE - 120:
            break
            
        s = _score(start_pts)
        if s > best_score:
            best_score = s
            best_points = start_pts.copy()
        
        # Optimize with Smooth Nelder-Mead
        if time.time() < _DEADLINE - 60:
            opt_pts = _nelder_mead_smooth_optimize(start_pts, max_iter=1500, alpha=50.0)
            s = _score(opt_pts)
            if s > best_score:
                best_score = s
                best_points = opt_pts.copy()
        
        # Second pass with higher alpha (sharper minimum) if time permits
        if best_points is not None and time.time() < _DEADLINE - 30:
            opt_pts2 = _nelder_mead_smooth_optimize(best_points, max_iter=1000, alpha=100.0)
            s = _score(opt_pts2)
            if s > best_score:
                best_score = s
                best_points = opt_pts2.copy()

    # Fallback
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = starting_configs[0]
    
    return best_points

# EVOLVE-BLOCK-END
