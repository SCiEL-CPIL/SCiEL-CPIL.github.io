# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull (vectorized)
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
        
    # Calculate all triangle areas efficiently using meshgrid approach
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
    
    Uses a combination of structured initial configurations and local search
    with random perturbations, inspired by high-scoring solutions.
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
    
    # Candidate 3: Two concentric polygons (7 and 6)
    def make_concentric_polygons(n1=7, n2=6, r1=1.0, r2=0.5):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        p1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        p2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([p1, p2])
    
    candidates.append(make_concentric_polygons(7, 6, 1.0, 0.5))
    candidates.append(make_concentric_polygons(6, 7, 1.0, 0.6))
    
    # Candidate 4: Fibonacci-like distribution
    def make_fibonacci_2d():
        n = 13
        golden_ratio = (1 + np.sqrt(5)) / 2
        angles = 2 * np.pi * np.arange(n) / golden_ratio
        radii = np.sqrt(np.arange(n) / n)
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return points
    
    candidates.append(make_fibonacci_2d())
    
    # Candidate 5: Grid-like arrangement
    def make_grid_arrangement():
        # 4x3 grid with one extra point
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0, 1, 2, 3])
        # Take 13 points from a 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                if len(points) < 13:
                    points.append([i, j])
        points = np.array(points)
        # Center and scale
        points = (points - points.mean(axis=0)) / points.std(axis=0)
        return points
    
    candidates.append(make_grid_arrangement())
    
    # Add random perturbations to structured configs
    rng = np.random.default_rng(12345)
    for base in candidates[:3]:
        perturb = rng.normal(0, 0.05, size=base.shape)
        candidates.append(base + perturb)
    
    # Add some fully random starts
    for _ in range(3):
        pts_rand = rng.random((13, 2)) * 2 - 1
        candidates.append(pts_rand)
    
    # Local search function
    def local_search(points, time_limit):
        points = points.copy()
        current_score = compute_score(points)
        best_score_local = current_score
        best_points_local = points.copy()
        
        step_size = 0.1
        no_improve_count = 0
        rng = np.random.default_rng(seed=int(time.time() * 1000) % 100000)
        
        while time.time() < time_limit:
            improved = False
            # Try moving each point with random perturbation
            for idx in range(13):
                # Generate a random perturbation
                dx = rng.uniform(-step_size, step_size)
                dy = rng.uniform(-step_size, step_size)
                
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
                # If we improved, keep step size or increase slightly
                pass
        
        return best_points_local, best_score_local
    
    # Run local search on each candidate
    n_candidates = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline:
            break
            
        # Allocate time for this candidate
        # Distribute remaining time among remaining candidates
        remaining_time = deadline - time.time()
        remaining_candidates = n_candidates - i
        candidate_time = remaining_time / remaining_candidates
        candidate_deadline = min(deadline, time.time() + candidate_time)
        
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
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0]
    
    # Ensure all points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]
    
    return best_points

# EVOLVE-BLOCK-END
