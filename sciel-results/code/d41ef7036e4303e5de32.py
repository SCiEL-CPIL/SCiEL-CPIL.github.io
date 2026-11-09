import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the score A_min / A_hull for a set of points.
    """
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
        
    # Compute all triangle areas vectorized
    # Points shape: (n, 2)
    # We need all combinations of 3 points
    # Using broadcasting to compute areas
    
    # Get indices for all combinations
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    
    # Filter to i < j < k to avoid duplicates and self-combinations
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    # Extract coordinates
    x1, y1 = points[i, 0], points[i, 1]
    x2, y2 = points[j, 0], points[j, 1]
    x3, y3 = points[k, 0], points[k, 1]
    
    # Area formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def _local_search(points: np.ndarray, start_time: float, max_time: float = 600.0, step_size: float = 0.05) -> np.ndarray:
    """
    Perform local search to maximize the score.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(current_points)
    rng = np.random.default_rng(seed=123)
    
    # Adaptive step size
    current_step = step_size
    
    while time.time() - start_time < max_time - 10:  # Leave 10s margin
        improved = False
        # Try moving each point in random directions
        for i in range(n):
            # Generate a few random perturbations for point i
            for _ in range(5):
                # Random direction
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, current_step)
                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)
                
                new_points = current_points.copy()
                new_points[i, 0] += dx
                new_points[i, 1] += dy
                
                new_score = _compute_score(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_points = current_points.copy()
                        best_score = current_score
                        # Optionally reduce step size slightly on improvement
                        current_step *= 0.95
                        if current_step < 0.001:
                            current_step = 0.001
                else:
                    # If no improvement, we might want to keep the step size or increase it slightly
                    pass
        
        if not improved:
            # If no improvement in a full pass, reduce step size
            current_step *= 0.9
            if current_step < 0.0001:
                break
                
    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points.
    """
    start_time = time.time()
    max_time = 600.0
    
    # Initialization: Regular 13-gon
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Initial score
    initial_score = _compute_score(points)
    
    # Run local search
    best_points = _local_search(points, start_time, max_time)
    best_score = _compute_score(best_points)
    
    # If we have time, try a few other initializations
    # For example, a regular 12-gon plus center? Or random?
    # Given the time budget, one good run is likely sufficient, 
    # but let's try a second run with a slightly perturbed regular polygon if time permits.
    
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 100:
        # Perturb the initial points slightly and run again
        rng = np.random.default_rng(seed=456)
        perturbed_points = points + rng.normal(0, 0.1, size=(n, 2))
        # Ensure they stay within a reasonable bound
        perturbed_points = np.clip(perturbed_points, -2, 2)
        
        second_best_points = _local_search(perturbed_points, time.time(), remaining_time - 10)
        second_best_score = _compute_score(second_best_points)
        
        if second_best_score > best_score:
            best_points = second_best_points
            best_score = second_best_score
            
    return best_points
