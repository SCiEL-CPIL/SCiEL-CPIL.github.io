import numpy as np
import time
from scipy.optimize import minimize

def compute_metric(points):
    """
    Computes the metric described in the task contract.
    points: (N, 2) array
    Returns: (metric, is_valid)
    """
    N = points.shape[0]
    if N != 20:
        return 0.0, False
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0, False

    # Normalization
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    
    normalized = centered / mean_dist
    
    # Check for repeated points (minimum pairwise distance)
    # Use distance matrix
    diff = normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    if min_pair_dist < 1e-6:
        return 0.0, False

    # Check strictly convex
    edges = np.roll(normalized, -1, axis=0) - normalized
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    # All cross products must be non-zero and same sign
    if np.any(cross == 0):
        return 0.0, False
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0, False
    
    # Compute vertex scores
    # dist_matrix has inf on diagonal, so we can just sort each row
    # But we need distances from i to j (j!=i)
    # dist_matrix[i] gives distances from i to all j.
    # We need to exclude i itself. Since diagonal is inf, it will be at the end of sorted array.
    # We only care about the first 19.
    
    min_score = np.inf
    
    for i in range(N):
        d_sorted = np.sort(dist_matrix[i, :])[:19] # Exclude self (inf)
        
        # Check for < 0.05
        if d_sorted[0] < 0.05:
            score = 0.0
        else:
            # Find k minimizing d_{k+1} - d_k
            diffs = np.diff(d_sorted)
            min_diff_idx = np.argmin(diffs)
            k = min_diff_idx
            D = (d_sorted[k] + d_sorted[k+1]) / 2.0
            
            if D < 0.1:
                score = 0.0
            else:
                # Select four distances
                if k == 0:
                    selected = d_sorted[0:4]
                elif k == 17:
                    selected = d_sorted[15:19]
                else:
                    selected = d_sorted[k-1:k+3]
                
                ratios = np.maximum(selected / D, D / selected)
                raw = np.mean(ratios)
                score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0, True # Early exit if any vertex is 0
    
    return min_score, True

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 50s buffer for safety
    
    n = 20
    best_score = -1.0
    best_points = None
    
    # Initial regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Evaluate initial
    init_score, valid = compute_metric(init_points)
    if valid:
        best_score = init_score
        best_points = init_points
    
    # Define objective function for optimizer
    # We want to maximize the min score, so minimize -score
    # Add penalty for invalidity
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = params.reshape(n, 2)
        score, valid = compute_metric(pts)
        if not valid:
            # Penalty for invalidity: large value + sum of squares to guide back
            return 1e6 + np.sum(pts**2)
        return -score
    
    # Try optimization with Powell method
    # Powell is good for non-smooth, non-differentiable functions
    try:
        # Run a few restarts with small noise
        for restart in range(5):
            if time.time() > deadline - 60:
                break
                
            if restart == 0:
                x0 = init_points.flatten()
            else:
                # Add small random noise
                noise = np.random.normal(0, 0.05, (n, 2))
                x0 = (init_points + noise).flatten()
            
            res = minimize(objective, x0, method='Powell',
                           options={'maxiter': 2000, 'maxfev': 10000, 'xtol': 1e-5, 'ftol': 1e-5})
            
            if time.time() > deadline - 10:
                break
                
            final_pts = res.x.reshape(n, 2)
            final_score, valid = compute_metric(final_pts)
            
            if valid and final_score > best_score:
                best_score = final_score
                best_points = final_pts
                
    except Exception:
        pass

    if best_points is None:
        return init_points
    
    # Final check
    final_score, valid = compute_metric(best_points)
    if not valid:
        return init_points
        
    return best_points
