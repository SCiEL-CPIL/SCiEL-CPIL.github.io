import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_score_fast(points):
    """
    Compute the minimum triangle area and hull area efficiently.
    Returns (min_area, hull_area).
    """
    n = len(points)
    if n < 3:
        return 0.0, 0.0
        
    # 1. Compute all triangle areas
    # points shape: (n, 2)
    # We need areas for all i < j < k
    # Broadcasting: 
    # i_idx: (n, 1, 1), j_idx: (1, n, 1), k_idx: (1, 1, n)
    # This creates an n x n x n array which is 13^3 = 2197, very small.
    i_idx = np.arange(n)[:, None, None]
    j_idx = np.arange(n)[None, :, None]
    k_idx = np.arange(n)[None, None, :]
    
    # Points indexed by i, j, k
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    # Or determinant form:
    # | x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j) | / 2
    
    # Using cross product of vectors (pj-pi) and (pk-pi)
    v1 = pj - pi
    v2 = pk - pi
    
    # Cross product in 2D is scalar: x1*y2 - y1*x2
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    areas = 0.5 * np.abs(cross)
    
    # We only care about i < j < k to avoid duplicates and zero areas from identical points
    # Create a mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0, 0.0
        
    min_area = np.min(valid_areas)
    
    # 2. Compute Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0, 0.0
        
    if hull_area <= 0:
        return 0.0, 0.0
        
    return min_area, hull_area

def _score(points):
    a_min, a_hull = _compute_score_fast(points)
    if a_hull <= 0 or not np.isfinite(a_min):
        return 0.0
    return a_min / a_hull

def _local_search(points, time_limit, rng):
    """
    Perform a local search to improve the score.
    We try small random perturbations of individual points.
    """
    best_points = points.copy()
    best_score = _score(best_points)
    start_time = time.time()
    
    n = len(best_points)
    step_size = 0.1
    accept_prob = 0.1 # Accept 10% of bad moves initially (simple SA-like)
    
    # Simple local search: try moving each point slightly
    # We will iterate until time limit or convergence
    while time.time() - start_time < time_limit:
        improved = False
        # Try a few random moves
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random point
            idx = rng.integers(0, n)
            
            # Pick a random direction and magnitude
            angle = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(-step_size, step_size)
            
            dx = mag * np.cos(angle)
            dy = mag * np.sin(angle)
            
            new_points = best_points.copy()
            new_points[idx, 0] += dx
            new_points[idx, 1] += dy
            
            new_score = _score(new_points)
            
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                improved = True
            elif rng.random() < accept_prob * np.exp(-(best_score - new_score) / 0.01):
                # Simple simulated annealing acceptance
                best_points = new_points
                best_score = new_score
                improved = True
                
        if not improved:
            # Reduce step size and acceptance probability
            step_size *= 0.8
            accept_prob *= 0.9
            if step_size < 1e-4:
                break
                
    return best_points, best_score

def _init_concentric():
    """Initialize with two concentric polygons."""
    # Outer 7-gon, Inner 6-gon
    n_outer = 7
    n_inner = 6
    
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)]) * 1.0
    
    # Rotate inner to interleave
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    pts_inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)]) * 0.5
    
    points = np.vstack([pts_outer, pts_inner])
    return points

def _init_jittered_grid():
    """Initialize with a jittered grid pattern."""
    # 3x3 grid + 4 extra? No, 13 points.
    # Maybe a 4x4 grid subset?
    # Let's try a hexagonal packing approximation.
    # Center point, 6 around it, 6 more around that?
    
    pts = []
    # Center
    pts.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        
    # Ring 2: 6 points
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        
    points = np.array(pts)
    # Add some jitter
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.05, points.shape)
    return points

def _init_random_convex():
    """Initialize with a random convex polygon."""
    n = 13
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    radii = rng.uniform(0.8, 1.0, n)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580 # Leave margin for overhead
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # List of initializers
    initializers = [
        _init_concentric,
        _init_jittered_grid,
    ]
    
    # Add some random convex polygons
    for _ in range(3):
        def _rand_init(rng=rng):
            angles = np.sort(rng.uniform(0, 2 * np.pi, 13))
            radii = rng.uniform(0.5, 1.0, 13)
            return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializers.append(_rand_init)

    # Run local search from each initialization
    for init_func in initializers:
        if time.time() > deadline - 30:
            break
            
        try:
            points = init_func()
            if points is None or len(points) != 13:
                continue
                
            # Ensure finite
            if not np.all(np.isfinite(points)):
                continue
                
            # Local search with time budget
            # Allocate time based on remaining time
            remaining = deadline - time.time()
            time_per_init = max(5.0, remaining / (len(initializers) + 1))
            
            pts, score = _local_search(points, time_per_init, rng)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception as e:
            continue

    # If we still have time, do a final polish on the best result found
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        pts, score = _local_search(best_points, remaining, rng)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
