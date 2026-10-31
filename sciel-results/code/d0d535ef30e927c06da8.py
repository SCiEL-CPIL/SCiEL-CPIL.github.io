"""Optimized program for Heilbronn convex regions - 13 points maximizing min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def compute_min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    # Vectorized computation for speed
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(points):
    """Score = min triangle area / hull area. Higher is better."""
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = compute_min_triangle_area(points)
    return min_area / hull_area

def negative_score(params):
    """Negative score for minimization."""
    points = params.reshape(N, 2)
    s = score_function(points)
    return -s

def generate_grid_pattern():
    """4x4 grid minus one corner (top-right), centered."""
    points = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 3:  # skip top-right
                continue
            points.append([i - 1.5, j - 1.5])
    return np.array(points)

def generate_circular_pattern():
    """13 points on a circle."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def generate_hexagonal_pattern():
    """Hexagonal-like arrangement."""
    # Center + 6 inner + 6 outer
    points = [[0, 0]]
    for i in range(6):
        angle = i * np.pi / 3
        points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
    return np.array(points)

def generate_random_pattern(seed):
    """Random points in unit square."""
    rng = np.random.RandomState(seed)
    return rng.rand(N, 2)

def optimize_from_start(start_points, deadline, max_iter=50):
    """Local optimization from a starting configuration."""
    start_time = time.time()
    current = start_points.copy()
    current_score = score_function(current)
    
    best_points = current.copy()
    best_score = current_score
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            negative_score,
            current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.success or result.fun < -current_score:
            new_points = result.x.reshape(N, 2)
            new_score = -result.fun
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
    except Exception:
        pass
    
    return best_points, best_score

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Generate multiple starting configurations
    start_configs = [
        generate_grid_pattern(),
        generate_circular_pattern(),
        generate_hexagonal_pattern(),
        generate_random_pattern(42),
        generate_random_pattern(123),
        generate_random_pattern(999),
    ]
    
    best_points = start_configs[0]
    best_score = score_function(best_points)
    
    # Try optimizing from each starting configuration
    for i, start in enumerate(start_configs):
        if time.time() > deadline:
            break
        
        # Scale to reasonable size
        start_scaled = start.copy()
        # Normalize to have reasonable spread
        center = start_scaled.mean(axis=0)
        start_scaled = start_scaled - center
        
        optimized, score = optimize_from_start(start_scaled, deadline)
        
        if score > best_score:
            best_score = score
            best_points = optimized
    
    # Final check
    final_score = score_function(best_points)
    if final_score > best_score:
        best_score = final_score
    
    return best_points
