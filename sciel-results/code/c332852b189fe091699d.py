import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Check for valid hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
        
    # Calculate all triangle areas efficiently
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area ratio.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: Regular 12-gon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    pts1 = np.vstack([np.array([0.0, 0.0]), pts1])
    candidates.append(pts1)
    
    # Candidate 2: Two concentric hexagons + center
    # Hexagon 1: radius 0.5, angles 0, 60, ...
    angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    pts_h1 = np.column_stack([0.5 * np.cos(angles1), 0.5 * np.sin(angles1)])
    # Hexagon 2: radius 1.0, angles 30, 90, ...
    angles2 = angles1 + np.pi/6
    pts_h2 = np.column_stack([1.0 * np.cos(angles2), 1.0 * np.sin(angles2)])
    pts2 = np.vstack([np.array([0.0, 0.0]), pts_h1, pts_h2])
    candidates.append(pts2)
    
    # Candidate 3: Random points with symmetry enforcement? 
    # Just add a few random ones to escape local minima
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts_rand = rng.random((13, 2))
        candidates.append(pts_rand)
    
    # Local search function
    def local_search(points, time_limit):
        points = points.copy()
        current_score = compute_score(points)
        best_score_local = current_score
        best_points_local = points.copy()
        
        step_size = 0.1
        no_improve_count = 0
        
        while time.time() < time_limit:
            improved = False
            # Try moving each point with random perturbation
            for idx in range(13):
                # Generate a random perturbation
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                
                new_points = points.copy()
                new_points[idx, 0] += dx
                new_points[idx, 1] += dy
                
                new_score = compute_score(new_points)
                
                if new_score > current_score:
                    points = new_points
                    current_score = new_score
                    improved = True
                    no_improve_count = 0
                    
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_points_local = points.copy()
                else:
                    no_improve_count += 1
            
            if not improved:
                step_size *= 0.95
                if step_size < 1e-6:
                    break
            else:
                # If we improved, maybe keep step size or increase slightly?
                # Usually decrease to converge.
                pass
        
        return best_points_local, best_score_local
    
    # Run local search on each candidate
    for init_pts in candidates:
        if time.time() >= deadline:
            break
            
        # Allocate time for this candidate
        # If many candidates, split time.
        # Let's just run until deadline for the first few, then shorter for others?
        # Simpler: Run each for a fixed short time, or until deadline if only one left.
        # Let's run each for 100 seconds max, or until deadline.
        candidate_deadline = min(deadline, time.time() + 100.0)
        
        res_pts, res_score = local_search(init_pts, candidate_deadline)
        
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts
            
    # If we have time left, do a final polish on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        # Run another round of local search on the best found
        res_pts, res_score = local_search(best_points, deadline - 10.0)
        if res_score > best_score:
            best_score = res_score
            best_points = res_pts

    return best_points
