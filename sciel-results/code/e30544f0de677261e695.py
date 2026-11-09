import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def compute_vertex_scores(points):
    """
    Compute the minimum vertex score for a given set of points.
    Returns the score (higher is better) and a validity flag.
    """
    n = len(points)
    # 1. Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0, False
    points = points / mean_dist

    # 2. Check convexity
    # Compute cross products of consecutive edges
    # v_i = points[i], v_{i+1} = points[(i+1)%n]
    # edge_i = v_{i+1} - v_i
    # cross_i = edge_i x edge_{i+1}
    edges = np.roll(points, -1, axis=0) - points
    cross_products = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    
    # Strictly convex: no repeated points, every turn nonzero and of one sign
    # Check for repeated points
    if np.any(np.linalg.norm(np.diff(np.vstack([points, points[0]]), axis=0), axis=1) < 1e-6):
        return 0.0, False
        
    # Check turns
    if np.any(np.abs(cross_products) < 1e-6):
        return 0.0, False
        
    # All cross products must have the same sign
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0, False
        
    # Total turning +-2*pi is guaranteed for simple convex polygon
    
    # 3. Compute vertex scores
    # Distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    
    min_score = 1.0
    for i in range(n):
        # Distances from v_i to others
        d = dist_matrix[i].copy()
        d[i] = np.inf # Exclude self
        
        # Sort distances
        d_sorted = np.sort(d)
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            return 0.0, False # Vertex scores 0, so min is 0
            
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(diffs)
        
        D = (d_sorted[k] + d_sorted[k+1]) / 2.0
        if D < 0.1:
            return 0.0, False # Vertex scores 0
            
        # Select 4 distances
        if k == 0:
            selected_d = d_sorted[0:4]
        elif k == 17:
            selected_d = d_sorted[15:19]
        else:
            selected_d = d_sorted[k-1:k+3]
            
        # raw = mean of max(d/D, D/d)
        ratios = np.maximum(selected_d / D, D / selected_d)
        raw = np.mean(ratios)
        
        vertex_score = min(1.0, 1.0 / raw)
        
        if vertex_score < min_score:
            min_score = vertex_score
            
    return min_score, True

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    # Initial guess: regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initial_points = np.column_stack((x, y))
    
    # Objective function to minimize: -score
    def objective(params):
        points = params.reshape(n, 2)
        score, valid = compute_vertex_scores(points)
        if not valid:
            return 1.0 # Penalize invalid shapes
        return -score
        
    # Optimize using Nelder-Mead
    # We might need to try multiple starts
    best_score = -1.0
    best_params = None
    
    # Start with regular polygon
    result = minimize(objective, initial_points.flatten(), method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if result.fun < -best_score:
        best_score = -result.fun
        best_params = result.x
        
    # Try a few random perturbations
    np.random.seed(42)
    for _ in range(10):
        noise = np.random.normal(0, 0.1, (n, 2))
        perturbed = initial_points + noise
        result = minimize(objective, perturbed.flatten(), method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        if result.fun < -best_score:
            best_score = -result.fun
            best_params = result.x
            
    if best_params is None:
        return initial_points
        
    return best_params.reshape(n, 2)
