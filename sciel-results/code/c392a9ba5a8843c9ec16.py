# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized operations."""
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
    
    # Vectorized triangle area calculation
    # Get all combinations of indices
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    # Flatten and filter unique combinations i < j < k
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    # Area = 0.5 * |cross|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _local_search(points: np.ndarray, max_iter=10000, initial_step=0.05, time_limit=None) -> np.ndarray:
    """Perform local search to improve the point arrangement."""
    current = points.copy()
    current_score = _score(current)
    n = current.shape[0]
    step_size = initial_step
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
            
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
            # Slightly decrease step size to refine
            step_size = max(step_size * 0.98, 0.001)
        else:
            # Slightly increase step size to explore
            step_size = min(step_size * 1.02, 0.2)
    
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
    deadline = start_time + 590  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Grid
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.01, 0.99)
    
    # 2. Hexagonal
    hex_pts = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1],
        [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
        [0.2, 0.7], [0.4, 0.7]
    ])
    
    # 3. Random seeds
    rng = np.random.default_rng(42)
    random_starts = [rng.random((13, 2)) for _ in range(5)]
    random_starts = [np.clip(p, 0.01, 0.99) for p in random_starts]
    
    starts = [grid, hex_pts] + random_starts
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Local search from best start
    if best_points is not None:
        improved = _local_search(best_points, max_iter=5000, initial_step=0.05, time_limit=deadline - 10)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Random restarts
    while time.time() < deadline - 20:
        pts = rng.random((13, 2))
        pts = np.clip(pts, 0.01, 0.99)
        
        # Quick check
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
        
        # Local search with limited time
        # Allow up to 10s per restart
        restart_deadline = min(time.time() + 10, deadline - 5)
        improved = _local_search(pts, max_iter=2000, initial_step=0.1, time_limit=restart_deadline)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    if best_points is None:
        best_points = np.random.default_rng(42).random((13, 2))
    
    return best_points

# EVOLVE-BLOCK-END
