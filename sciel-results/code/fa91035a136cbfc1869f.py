"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0  # seconds


def _triangle_area(p1, p2, p3):
    """Area of triangle given three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    if n < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    return -_score(points)


def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _perturb(points, rng, scale=0.1):
    """Perturb points slightly."""
    return points + rng.uniform(-scale, scale, size=points.shape)


def _optimize_from_start(start_points, rng, deadline):
    """Run Nelder-Mead optimization from a starting point. Returns best found."""
    start = np.asarray(start_points, dtype=np.float64).reshape(-1, 2)
    # Flatten for optimizer
    x0 = start.flatten()
    
    best_x = x0.copy()
    best_val = _neg_score(x0.reshape(-1, 2))
    
    # Nelder-Mead with reasonable settings
    result = minimize(
        _neg_score,
        x0,
        method='Nelder-Mead',
        options={
            'xatol': 1e-6,
            'fatol': 1e-8,
            'maxiter': 200,
            'maxfev': 500,
            'adaptive': True,
        }
    )
    
    if result.fun < best_val:
        best_val = result.fun
        best_x = result.x
    
    # Check time
    if time.time() > deadline - SAFETY_MARGIN:
        return best_x.reshape(-1, 2)
    
    # Try a second round with tighter tolerance
    result2 = minimize(
        _neg_score,
        best_x,
        method='Nelder-Mead',
        options={
            'xatol': 1e-8,
            'fatol': 1e-10,
            'maxiter': 100,
            'maxfev': 300,
            'adaptive': True,
        }
    )
    
    if result2.fun < best_val:
        best_val = result2.fun
        best_x = result2.x
    
    return best_x.reshape(-1, 2)


def construct():
    """Construct 14 points maximizing min triangle area / hull area."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = []
    
    # Regular polygon
    starts.append(_regular_polygon(N))
    
    # Slightly perturbed regular polygons
    for scale in [0.05, 0.1, 0.2, 0.3]:
        starts.append(_perturb(_regular_polygon(N), rng, scale))
    
    # Random points in unit circle
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.3, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Sort by angle for consistent ordering
        order = np.argsort(angles)
        starts.append(pts[order])
    
    # Points on unit circle with random angular spacing
    for _ in range(3):
        gaps = rng.uniform(0.5, 1.5, N)
        gaps /= gaps.sum() * 2 * np.pi
        angles = np.cumsum(gaps)
        angles[0] = rng.uniform(0, 2 * np.pi)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        starts.append(pts)
    
    # Square-like distributions
    for _ in range(3):
        pts = rng.uniform(-1, 1, (N, 2))
        starts.append(pts)
    
    # Track best overall
    best_points = starts[0]
    best_score = _score(best_points)
    
    for i, start in enumerate(starts):
        if time.time() > deadline - SAFETY_MARGIN:
            break
        
        optimized = _optimize_from_start(start, rng, deadline)
        sc = _score(optimized)
        
        if sc > best_score:
            best_score = sc
            best_points = optimized
    
    # If we still have time, do one more polish on the best
    if time.time() < deadline - SAFETY_MARGIN:
        final_opt = _optimize_from_start(best_points, rng, deadline)
        final_sc = _score(final_opt)
        if final_sc > best_score:
            best_points = final_opt
            best_score = final_sc
    
    return np.asarray(best_points, dtype=np.float64)
