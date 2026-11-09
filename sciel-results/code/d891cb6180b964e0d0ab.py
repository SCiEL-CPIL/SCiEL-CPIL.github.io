# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _triangle_areas_vectorized(points: np.ndarray) -> np.ndarray:
    """
    Compute areas of all triangles formed by triples of points.
    Returns array of shape (n, n, n) where entry [i,j,k] is the area of triangle (i,j,k).
    We only care about i < j < k, but vectorized computation is fast.
    """
    n = points.shape[0]
    # Use the cross product formula: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Vectorized: for all i, j, k
    # p_i, p_j, p_k as (n, n, 2) arrays
    pi = points[:, np.newaxis, np.newaxis, :]  # (n, 1, 1, 2)
    pj = points[np.newaxis, :, np.newaxis, :]  # (1, n, 1, 2)
    pk = points[np.newaxis, np.newaxis, :, :]  # (1, 1, n, 2)
    
    # Cross product in 2D: (pj - pi) x (pk - pi) = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    v1 = pj - pi  # (n, n, 1, 2)
    v2 = pk - pi  # (n, 1, n, 2)
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # (n, n, n)
    areas = np.abs(cross) / 2.0
    return areas


def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
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
    areas = _triangle_areas_vectorized(points)
    
    # We need min over i < j < k
    # Create mask for i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    min_area = np.min(areas[mask])
    
    return min_area / A_hull


def _score_and_worst(points: np.ndarray) -> tuple:
    """
    Compute score and return (score, worst_triangle_indices).
    worst_triangle_indices is (i, j, k) with i < j < k for the minimum area triangle.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0, (0, 1, 2)
    
    if not np.all(np.isfinite(points)):
        return 0.0, (0, 1, 2)
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0, (0, 1, 2)
    
    if A_hull <= 0:
        return 0.0, (0, 1, 2)
    
    areas = _triangle_areas_vectorized(points)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    # Find min area among valid triples
    masked_areas = np.where(mask, areas, np.inf)
    flat_min_idx = np.argmin(masked_areas)
    i, j, k = np.unravel_index(flat_min_idx, (n, n, n))
    min_area = areas[i, j, k]
    
    score = min_area / A_hull
    return score, (i, j, k)


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    
    Strategy:
    1. Start with a regular 13-gon.
    2. Use simulated annealing with local search targeting the worst triangle.
    3. Vectorized score computation for speed.
    4. Deadline guard to stay within time limits.
    """
    n = 13
    deadline = time.time() + 600.0  # 600 second budget
    
    # Start with regular 13-gon centered at origin
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = _score(best_points)
    
    # Simulated annealing parameters
    T = 0.1  # initial temperature
    cooling = 0.9999  # cooling rate
    min_T = 1e-6
    max_iter = 200000
    
    rng = np.random.default_rng(seed=12345)
    
    current_points = best_points.copy()
    current_score = best_score
    
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        if current_score <= 0:
            break
        
        # Identify the worst triangle
        score, (wi, wj, wk) = _score_and_worst(current_points)
        if score != current_score:
            # Score changed, update
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
            current_score = score
        
        # Perturb points in the worst triangle
        # Move each point in the worst triangle slightly
        step_size = T * 0.1
        perturb = rng.normal(0, step_size, size=(3, 2))
        
        new_points = current_points.copy()
        new_points[wi] += perturb[0]
        new_points[wj] += perturb[1]
        new_points[wk] += perturb[2]
        
        # Also perturb a random other point slightly for global exploration
        if rng.random() < 0.3:
            other_idx = rng.integers(0, n)
            if other_idx not in (wi, wj, wk):
                new_points[other_idx] += rng.normal(0, T * 0.05, size=2)
        
        new_score = _score(new_points)
        
        # Accept if better, or with probability exp((new - old)/T) if worse
        if new_score > current_score or (new_score > 0 and np.log(rng.random()) < (new_score - current_score) / T):
            current_points = new_points
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Final local refinement: gradient-free search targeting worst triangles
    for refine_iter in range(5000):
        if time.time() > deadline - 10:
            break
        
        score, (wi, wj, wk) = _score_and_worst(best_points)
        if score <= 0:
            break
        
        # Try small perturbations in multiple directions
        best_trial = None
        best_trial_score = score
        
        for trial in range(20):
            step = 0.01 * (1.0 + 0.5 * np.random.random())
            directions = np.random.choice([-1, 1], size=(3, 2))
            new_points = best_points.copy()
            new_points[wi] += step * directions[0]
            new_points[wj] += step * directions[1]
            new_points[wk] += step * directions[2]
            
            trial_score = _score(new_points)
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial = new_points.copy()
        
        if best_trial is not None:
            best_points = best_trial
            best_score = best_trial_score
    
    return best_points


# EVOLVE-BLOCK-END
