"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle using cross product
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area formula on convex hull points
        try:
            hull = ConvexHull(points)
            pts = points[hull.vertices]
            n = len(pts)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
            return abs(area) / 2.0
        except Exception:
            return 1.0


def _objective(params):
    """Objective function to minimize (negative of score)."""
    points = params.reshape(-1, 2)
    # Ensure points are within a reasonable range
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1.0
    
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 1.0
    
    min_tri = _min_triangle_area(points)
    score = min_tri / hull_area
    return -score  # We minimize negative score to maximize score


def construct():
    """Construct 13 points to maximize minimum triangle area / convex hull area."""
    # Define bounds for each coordinate
    bounds = [(-1, 1)] * (2 * N)
    
    # Use differential evolution for global optimization
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        popsize=15,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True,
        init='sobol'
    )
    
    best_points = result.x.reshape(-1, 2)
    
    # Also try a few other starting points for robustness
    best_score = -result.fun
    best_result = best_points
    
    # Try additional random restarts
    np_random = np.random.RandomState(42)
    for _ in range(5):
        init_points = np_random.uniform(-1, 1, size=(2 * N,))
        res = differential_evolution(
            _objective,
            bounds,
            x0=init_points,
            seed=42 + _,
            maxiter=50,
            tol=1e-6,
            popsize=10,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol'
        )
        if -res.fun > best_score:
            best_score = -res.fun
            best_result = res.x.reshape(-1, 2)
    
    # Normalize the points: center them and scale to unit area
    # First, center
    best_result = best_result - best_result.mean(axis=0)
    
    # Scale to unit convex hull area
    hull_area = _convex_hull_area(best_result)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_result = best_result * scale
    
    return best_result
# EVOLVE-BLOCK-END
