# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from itertools import combinations

def _score_fast(points):
    """Compute A_min / A_hull for a set of points using vectorized operations."""
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    n = len(points)
    # Vectorized triangle area computation
    # For each triple (i,j,k), area = |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)| / 2
    # We can compute this more efficiently by iterating over pairs and computing cross products
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            # Vector: points[j] - points[i]
            v1 = points[j] - points[i]
            # For all k > j, compute area of triangle (i, j, k)
            # area = 0.5 * |cross(v1, points[k] - points[i])|
            for k in range(j + 1, n):
                v2 = points[k] - points[i]
                area = abs(v1[0] * v2[1] - v1[1] * v2[0]) * 0.5
                if area < min_area:
                    min_area = area
    
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _score(points):
    """Compute A_min / A_hull for a set of points."""
    return _score_fast(points)

def _score_batched(points, top_k=50):
    """
    Compute score but focus on the smallest triangles using Delaunay triangulation.
    The minimum triangle area in a point set is always among the Delaunay triangles
    or triangles formed by nearby points.
    """
    if len(points) < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    n = len(points)
    # Compute all triangle areas vectorized
    # Create index arrays
    indices = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(indices, indices, indices, indexing='ij')
    i_flat = i_idx.flatten()
    j_flat = j_idx.flatten()
    k_flat = k_idx.flatten()
    
    # Only keep unique triples i < j < k
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_flat = i_flat[mask]
    j_flat = j_flat[mask]
    k_flat = k_flat[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 13
    
    # Best configuration found so far
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Try multiple initializations
    initializations = []
    
    # 1. Hexagonal lattice pattern
    # For 13 points, use a hexagonal arrangement
    # Center + 6 at distance 1 + 6 at distance sqrt(3)
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initializations.append(reg_pts)
    
    # 3. Grid-based
    grid_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Random points in unit square
    rng = np.random.default_rng(seed=12345)
    for _ in range(3):
        initializations.append(rng.random((n, 2)))
    
    # Evaluate initializations
    for init_pts in initializations:
        if time.time() > deadline - 30:
            break
        s = _score(init_pts)
        update_best(init_pts, s)
    
    # Local optimization using coordinate descent with adaptive steps
    current_points = best_points.copy()
    current_score = best_score
    
    # Use a more sophisticated local search
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for step_idx, base_step in enumerate(step_sizes):
        if time.time() > deadline - 30:
            break
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                # Try multiple directions
                n_tries = 30
                for _ in range(n_tries):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    
                    # Try a few step sizes around base_step
                    for mult in [1.0, 0.5, 0.25, 2.0]:
                        step = base_step * mult
                        trial_points = current_points.copy()
                        trial_points[i] = current_points[i] + step * direction
                        s = _score(trial_points)
                        if s > current_score:
                            current_points = trial_points
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            if time.time() > deadline - 30:
                break
        
        update_best(current_points, current_score)
    
    # Try a few random restarts from perturbed best
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        perturbed = best_points + rng.normal(size=(n, 2)) * 0.1
        s = _score(perturbed)
        if s > best_score:
            best_points = perturbed.copy()
            best_score = s
        
        # Quick local optimization from here
        current = best_points.copy()
        cur_score = best_score
        for i in range(n):
            if time.time() > deadline - 30:
                break
            for _ in range(20):
                direction = rng.normal(size=2)
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction /= norm
                for step in [0.05, 0.02, 0.01]:
                    trial = current.copy()
                    trial[i] = current[i] + step * direction
                    s = _score(trial)
                    if s > cur_score:
                        current = trial
                        cur_score = s
                        break
        if cur_score > best_score:
            best_points = current.copy()
            best_score = cur_score
    
    # Final validation
    final_score = _score(best_points)
    if final_score <= 0 or not np.all(np.isfinite(best_points)):
        # Fallback
        fallback = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
