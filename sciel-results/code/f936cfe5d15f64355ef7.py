# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
                if area < min_area:
                    min_area = area
    return min_area


def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Compute the score: A_min / A_hull."""
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    if a_hull <= 1e-12 or a_min <= 1e-12:
        return 0.0
    return a_min / a_hull


def _hexagonal_lattice_13():
    """
    Create a hexagonal lattice arrangement of 13 points.
    Center + 6 in inner ring + 6 in outer ring.
    """
    points = []
    # Center
    points.append([0.0, 0.0])
    # Inner ring: 6 points at radius r1
    r1 = 1.0
    for i in range(6):
        angle = i * np.pi / 3.0
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    # Outer ring: 6 points at radius r2, offset by 30 degrees
    r2 = 2.0
    for i in range(6):
        angle = i * np.pi / 3.0 + np.pi / 6.0
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(points)


def _regular_polygon_13():
    """Create 13 points on a regular polygon (one at center, 12 on circle)."""
    points = []
    points.append([0.0, 0.0])
    r = 1.0
    for i in range(12):
        angle = i * 2 * np.pi / 12.0
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _triangular_grid_13():
    """Create 13 points in a triangular grid pattern."""
    points = []
    # Row 0: 1 point
    points.append([0.0, 0.0])
    # Row 1: 3 points
    dy = np.sqrt(3) / 2
    for i in range(3):
        points.append([(i - 1) * 1.0, dy])
    # Row 2: 4 points
    for i in range(4):
        points.append([(i - 1.5) * 1.0, 2 * dy])
    # Row 3: 5 points
    for i in range(5):
        points.append([(i - 2.0) * 1.0, 3 * dy])
    return np.array(points[:13])


def _optimize_from_start(points_start, time_limit):
    """
    Optimize starting from points_start using Nelder-Mead.
    Returns the best points found.
    """
    start_time = time.time()
    n = len(points_start)
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        return -_score(pts)
    
    x0 = points_start.flatten()
    
    best_points = points_start.copy()
    best_score = _score(points_start)
    
    # Run Nelder-Mead with time check
    result = minimize(
        objective, x0, method='Nelder-Mead',
        options={
            'maxiter': 500,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    if result.success or result.fun < -best_score:
        candidate = result.x.reshape(n, 2)
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_points = candidate
    
    return best_points, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses hexagonal lattice seeds with gradient-free optimization.
    """
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    # Multiple diverse starting configurations
    starts = [
        _hexagonal_lattice_13(),
        _regular_polygon_13(),
        _triangular_grid_13(),
    ]
    
    # Add some perturbed versions
    rng = np.random.default_rng(seed=42)
    for base in starts:
        for scale in [0.8, 1.2]:
            perturbed = base * scale + rng.normal(0, 0.05, base.shape)
            starts.append(perturbed)
    
    # Also try a more spread hexagonal pattern
    hex_spread = _hexagonal_lattice_13() * 1.5
    starts.append(hex_spread)
    
    best_points = starts[0]
    best_score = _score(best_points)
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 60:  # Need at least 60s for one more optimization
            break
        
        # Optimize from this start
        points_opt, score_opt = _optimize_from_start(start, time_limit=50)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = points_opt
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Final verification
    if not np.all(np.isfinite(best_points)):
        best_points = _hexagonal_lattice_13()
    
    return best_points


# EVOLVE-BLOCK-END
