# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _compute_scores(points: np.ndarray) -> np.ndarray:
    """
    Compute the vertex score for all vertices efficiently.
    Returns an array of 20 scores.
    """
    n = len(points)
    # Calculate pairwise distances
    # dists[i, j] is distance between point i and point j
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    
    scores = np.zeros(n)
    for i in range(n):
        # Get distances from i to all others, excluding self
        d = dists[i].copy()
        d[i] = np.inf
        d = np.sort(d)
        
        if d[0] < 0.05:
            scores[i] = 0.0
            continue
            
        gaps = np.diff(d)
        k = int(np.argmin(gaps))
        D = (d[k] + d[k + 1]) / 2.0
        
        if D < 0.1:
            scores[i] = 0.0
            continue
            
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k - 1:k + 3]
            
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        scores[i] = min(1.0, 1.0 / raw)
        
    return scores

def _min_vertex_score(points: np.ndarray) -> float:
    """Compute the minimum vertex score over all vertices."""
    scores = _compute_scores(points)
    return np.min(scores)

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized to maximize the minimum vertex equidistance score.
    Uses Differential Evolution to optimize angular positions.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    def objective(params: np.ndarray) -> float:
        # params: 20 angular perturbations
        angles = base_angles + params
        # Ensure angles are sorted to maintain cyclic order and convexity
        angles = np.sort(angles)
        
        # Check for degenerate cases (points too close)
        # We add 2pi to the first angle to check the wrap-around gap
        extended_angles = np.append(angles, angles[0] + 2.0 * np.pi)
        gaps = np.diff(extended_angles)
        if np.min(gaps) < 0.05: # Threshold to ensure distinct points and strict convexity
            return 1.0 # Maximize negative score, so return high penalty (low score)
            
        x = np.cos(angles)
        y = np.sin(angles)
        points = np.stack([x, y], axis=1)
        
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(points, axis=0)
        points = points - centroid
        dists_to_centroid = np.linalg.norm(points, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 1.0
            
        points = points / mean_dist
        
        # Check convexity: all cross products should have the same sign
        # Since points are on a circle (unit radius after normalization? No, they are on unit circle before normalization, so they are convex if angles are sorted and distinct).
        # Wait, points are on the unit circle before normalization. Any subset of points on a circle in sorted angular order forms a convex polygon.
        # So strict convexity is guaranteed by sorted distinct angles on a circle.
        # We just need to ensure no repeated points.
        
        score = _min_vertex_score(points)
        return -score # We want to minimize negative score

    # Bounds for the angular perturbations
    # We allow perturbations up to +/- pi/4 (45 degrees) to allow significant shape changes
    # while maintaining some structure. The sorting will handle the order.
    bounds = [(-0.5 * np.pi, 0.5 * np.pi) for _ in range(n)]
    
    # Run Differential Evolution
    # maxiter is limited to ensure we finish in time.
    # tol and atol control convergence.
    result = differential_evolution(objective, bounds, 
                                    maxiter=200, 
                                    popsize=20, 
                                    tol=1e-4,
                                    atol=1e-4,
                                    seed=42,
                                    polish=False) # Disable polish to save time, DE is usually good enough
    
    best_params = result.x
    best_angles = np.sort(base_angles + best_params)
    
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.stack([x, y], axis=1)
    
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist > 0.01:
        points = points / mean_dist
        
    return points

# EVOLVE-BLOCK-END
