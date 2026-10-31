# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area_vec(points):
    """
    Compute the minimum triangle area over all triples of points using vectorization.
    points: shape (N, 2)
    """
    n = len(points)
    # Generate indices for all unique triples (i, j, k) with i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask].flatten()
    j_idx = j_idx[mask].flatten()
    k_idx = k_idx[mask].flatten()
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area_fast(points):
    """
    Fast convex hull area calculation assuming points are in convex position.
    Falls back to scipy if not convex.
    """
    n = len(points)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    v1 = sorted_points[1:] - sorted_points[:-1]
    v2 = sorted_points[2:] - sorted_points[1:-1]
    v1_last = sorted_points[0] - sorted_points[-1]
    v2_last = sorted_points[1] - sorted_points[0]
    
    cross_2d = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    cross_last = v1_last[0] * v2_last[1] - v1_last[1] * v2_last[0]
    
    if np.all(cross_2d > 1e-9) and cross_last > 1e-9:
        pass
    elif np.all(cross_2d < -1e-9) and cross_last < -1e-9:
        pass
    else:
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 1e-10
            
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area_vec(points)
    ha = _convex_hull_area_fast(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    Approach: Simulated Annealing with vectorized evaluation and fast convex hull check.
    """
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    p = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_p = p.copy()
    best_score = _score(p)
    score_old = best_score
    
    T_start = 0.05
    cooling_rate = 0.999
    max_iters = 5000
    start_time = time.time()
    timeout = 110.0
    
    np.random.seed(42)
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        noise = np.random.normal(0, T_start * 0.5, (N, 2))
        p_new = p + noise
        
        score_new = _score(p_new)
        
        if score_new > score_old:
            p = p_new
            score_old = score_new
            if score_new > best_score:
                best_score = score_new
                best_p = p_new.copy()
        else:
            delta = score_old - score_new
            if delta > 0:
                prob = np.exp(-delta / (T_start + 1e-12))
                if np.random.rand() < prob:
                    p = p_new
                    score_old = score_new
            
        T_start *= cooling_rate
        
    T_polish = 1e-4
    for i in range(1000):
        if time.time() - start_time > timeout:
            break
        noise = np.random.normal(0, T_polish * 0.1, (N, 2))
        p_new = p + noise
        score_new = _score(p_new)
        if score_new > score_old:
            p = p_new
            score_old = score_new
            if score_new > best_score:
                best_score = score_new
                best_p = p_new.copy()
        T_polish *= 0.99

    hull_area = _convex_hull_area_fast(best_p)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_p = best_p * scale
        
    return best_p
# EVOLVE-BLOCK-END
