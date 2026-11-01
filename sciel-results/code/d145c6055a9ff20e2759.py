"""Heilbronn convex regions n=14: maximize min triangle area / convex hull area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0  # seconds before limit to stop


def _triangle_areas(points):
    """Compute all C(n,3) triangle areas for n points. Returns array of shape (C(n,3),)."""
    n = len(points)
    # Vectorized computation
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p1 = points[i_idx]
    p2 = points[j_idx]
    p3 = points[k_idx]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    return 0.5 * np.abs(cross)


def _convex_hull_area(points):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    
    points_sorted = np.sort(points, axis=0)
    points_sorted = points_sorted[np.lexsort((points_sorted[:, 0], points_sorted[:, 1]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in points_sorted:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points_sorted):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    tri_areas = _triangle_areas(points)
    min_tri = np.min(tri_areas)
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    return min_tri / hull_area


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    best_points = None
    best_score = -1.0
    
    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s
    
    # Initial good starting point: regular heptagon + inner heptagon
    def make_initial():
        outer = np.array([[np.cos(2 * np.pi * i / 7), np.sin(2 * np.pi * i / 7)] for i in range(7)])
        inner = 0.5 * np.array([[np.cos(2 * np.pi * i / 7 + np.pi / 7), np.sin(2 * np.pi * i / 7 + np.pi / 7)] for i in range(7)])
        return np.vstack([outer, inner])
    
    init = make_initial()
    update_best(init)
    
    # Strategy 1: Differential evolution with parameterized structure
    # Parameters: 14*2 = 28 coordinates, but we'll use a structured parameterization
    # Actually, let's just optimize all 28 coordinates directly
    
    def objective(params):
        pts = params.reshape(14, 2)
        # Normalize to unit area for stability
        hull_a = _convex_hull_area(pts)
        if hull_a < 1e-10:
            return 1e6
        pts_norm = pts / np.sqrt(hull_a)
        return -_score(pts_norm)
    
    # Run differential evolution
    try:
        result_de = differential_evolution(
            objective,
            bounds=[(-2, 2)] * 28,
            seed=42,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='sobol',
            callback=None
        )
        
        if result_de.success or result_de.fun < 1e5:
            pts_de = result_de.x.reshape(14, 2)
            # Normalize
            hull_a = _convex_hull_area(pts_de)
            if hull_a > 1e-10:
                pts_de = pts_de / np.sqrt(hull_a)
            update_best(pts_de)
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 2: Local refinement from multiple starting points
    start_points = [
        make_initial(),
        np.array([[np.cos(2 * np.pi * i / 14), np.sin(2 * np.pi * i / 14)] for i in range(14)]),
        np.array([[np.cos(2 * np.pi * i / 7), np.sin(2 * np.pi * i / 7)] for i in range(7)] +
                 [[0.3 * np.cos(2 * np.pi * i / 7 + np.pi / 7), 0.3 * np.sin(2 * np.pi * i / 7 + np.pi / 7)] for i in range(7)]),
        init + np.random.RandomState(0).normal(0, 0.1, (14, 2)),
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        # Normalize
        hull_a = _convex_hull_area(sp)
        if hull_a > 1e-10:
            sp = sp / np.sqrt(hull_a)
        
        def local_obj(params):
            pts = params.reshape(14, 2)
            hull_a = _convex_hull_area(pts)
            if hull_a < 1e-10:
                return 1e6
            return -_score(pts / np.sqrt(hull_a))
        
        try:
            result_min = minimize(
                local_obj,
                sp.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            pts_local = result_min.x.reshape(14, 2)
            hull_a = _convex_hull_area(pts_local)
            if hull_a > 1e-10:
                pts_local = pts_local / np.sqrt(hull_a)
            update_best(pts_local)
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 3: More differential evolution runs with different seeds
    for seed in [0, 1, 2, 3, 4]:
        if time.time() > deadline:
            break
        
        try:
            result_de2 = differential_evolution(
                objective,
                bounds=[(-2, 2)] * 28,
                seed=seed,
                maxiter=100,
                popsize=12,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=True,
                init='latinhypercube'
            )
            
            if result_de2.fun < 1e5:
                pts_de2 = result_de2.x.reshape(14, 2)
                hull_a = _convex_hull_area(pts_de2)
                if hull_a > 1e-10:
                    pts_de2 = pts_de2 / np.sqrt(hull_a)
                update_best(pts_de2)
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_points
    
    # Strategy 4: Final high-precision polish
    if best_points is not None:
        def final_obj(params):
            pts = params.reshape(14, 2)
            hull_a = _convex_hull_area(pts)
            if hull_a < 1e-10:
                return 1e6
            return -_score(pts / np.sqrt(hull_a))
        
        try:
            result_final = minimize(
                final_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            pts_final = result_final.x.reshape(14, 2)
            hull_a = _convex_hull_area(pts_final)
            if hull_a > 1e-10:
                pts_final = pts_final / np.sqrt(hull_a)
            update_best(pts_final)
        except Exception:
            pass
    
    return best_points
