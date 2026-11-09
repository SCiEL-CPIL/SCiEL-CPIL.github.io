# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools


def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Compute all triangle areas and find the minimum
    min_area = float('inf')
    for i, j, k in itertools.combinations(range(n), 3):
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
        if area < min_area:
            min_area = area
            if min_area < 1e-15:
                return 0.0
    
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iter=5000, step_size=0.01) -> np.ndarray:
    """Perform local search to improve the point arrangement."""
    current = points.copy()
    current_score = _score(current)
    n = current.shape[0]
    
    for iteration in range(max_iter):
        # Pick a random point to move
        idx = np.random.randint(n)
        # Generate a small random perturbation
        delta = np.random.normal(0, step_size, 2)
        new_point = current[idx] + delta
        
        # Keep points within [0, 1]^2
        if new_point[0] < 0 or new_point[0] > 1 or new_point[1] < 0 or new_point[1] > 1:
            continue
        
        candidate = current.copy()
        candidate[idx] = new_point
        candidate_score = _score(candidate)
        
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            step_size = max(step_size * 0.99, 0.001)
        else:
            step_size = min(step_size * 1.01, 0.1)
    
    return current


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Start with a regular grid-like arrangement
    best_points = None
    best_score = -1.0
    
    # Try different initial configurations
    configs = [
        # Regular grid 4x4 minus one corner
        lambda: np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)]),
        # Hexagonal-like arrangement
        lambda: np.array([
            [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1],
            [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
            [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
            [0.2, 0.7], [0.4, 0.7]
        ]),
        # Random
        lambda: np.random.default_rng(42).random((13, 2)),
    ]
    
    for config_func in configs:
        if time.time() > deadline - 30:
            break
        
        points = config_func()
        # Ensure points are in [0,1]^2
        points = np.clip(points, 0.01, 0.99)
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Local search
        improved = _local_search(points, max_iter=2000, step_size=0.05)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Additional random restarts if time permits
    rng = np.random.default_rng(123)
    restarts = 0
    while time.time() < deadline - 60 and restarts < 20:
        points = rng.random((13, 2))
        points = np.clip(points, 0.01, 0.99)
        
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        improved = _local_search(points, max_iter=1000, step_size=0.05)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
        
        restarts += 1
    
    if best_points is None:
        # Fallback to random
        best_points = np.random.default_rng(42).random((13, 2))
    
    return best_points


# EVOLVE-BLOCK-END
