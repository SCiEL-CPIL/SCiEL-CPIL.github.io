"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: compute area of convex hull manually
        # Sort points by angle from centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_points = points[np.argsort(angles)]
        # Shoelace formula
        n = len(sorted_points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += sorted_points[i, 0] * sorted_points[j, 1]
            area -= sorted_points[j, 0] * sorted_points[i, 1]
        return abs(area) / 2.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _make_two_shell(m, k, r_inner, angle_offset):
    """
    Create points: m points on outer circle (radius 1), k points on inner circle (radius r_inner).
    angle_offset: rotation of inner ring relative to outer ring.
    """
    outer_angles = 2 * np.pi * np.arange(m) / m
    inner_angles = 2 * np.pi * np.arange(k) / k + angle_offset
    
    outer_points = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner_points = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
    
    return np.vstack([outer_points, inner_points])

def _optimize_two_shell(m, k, start_r=0.5, start_angle=0.0, max_iter=2000):
    """Optimize (r_inner, angle_offset) for given partition (m, k)."""
    def objective(params):
        r_inner, angle_offset = params
        if r_inner < 0.01 or r_inner > 0.99:
            return 0.0
        points = _make_two_shell(m, k, r_inner, angle_offset)
        return -_score(points)  # Minimize negative score
    
    x0 = np.array([start_r, start_angle])
    bounds = [(0.01, 0.99), (-np.pi, np.pi)]
    
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.success:
        r_opt, angle_opt = result.x
        points = _make_two_shell(m, k, r_opt, angle_opt)
        return points, _score(points)
    else:
        return None, 0.0

def construct():
    deadline = time.time() + 100  # 100s safety margin within 120s limit
    
    best_points = None
    best_score = 0.0
    
    # Try all partitions m + k = 13 where m >= k and both >= 3
    partitions = []
    for m in range(3, 11):  # m from 3 to 10
        k = 13 - m
        if k >= 3:
            partitions.append((m, k))
    
    # Also try the regular 13-gon as baseline
    angles = 2 * np.pi * np.arange(N) / N
    regular_points = np.column_stack((np.cos(angles), np.sin(angles)))
    regular_score = _score(regular_points)
    if regular_score > best_score:
        best_score = regular_score
        best_points = regular_points.copy()
    
    # Try two-shell configurations for each partition
    for m, k in partitions:
        if time.time() > deadline:
            break
        
        # Try multiple starting points
        for start_r in [0.3, 0.5, 0.7]:
            for start_angle in [0.0, np.pi / m, np.pi / k]:
                if time.time() > deadline:
                    break
                
                points, score = _optimize_two_shell(m, k, start_r, start_angle, max_iter=500)
                if points is not None and score > best_score:
                    best_score = score
                    best_points = points.copy()
        
        # If we found a good solution, do a finer local search
        if best_score > 0:
            if time.time() > deadline:
                break
            # Use the best found as starting point for finer optimization
            # Extract r and angle from best_points if it matches this partition
            # For now, just do one more optimization with tighter tolerance
            points, score = _optimize_two_shell(m, k, 0.5, 0.0, max_iter=1000)
            if points is not None and score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Final fallback: if no improvement, return regular polygon
    if best_points is None:
        return regular_points
    
    return best_points
# EVOLVE-BLOCK-END
