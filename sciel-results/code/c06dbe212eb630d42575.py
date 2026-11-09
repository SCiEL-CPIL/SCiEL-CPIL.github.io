import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial guess: Regular polygon
    radii = np.ones(n)
    
    # Define the scoring function
    def calc_score(radii):
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Normalize
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        dists_to_centroid = np.linalg.norm(pts, axis=1)
        mean_dist = np.mean(dists_to_centroid)
        if mean_dist <= 0.01:
            return 0.0
        pts = pts / mean_dist
        
        # Check convexity (simplified: all cross products same sign)
        # And no repeated points
        # For small perturbations of regular polygon, convexity holds.
        # We'll assume it holds for the search, but verify at the end.
        
        min_vertex_score = 1.0
        for i in range(n):
            v = pts[i]
            # Distances to all other points
            dists = np.linalg.norm(pts - v, axis=1)
            # Remove self distance (0)
            dists = np.delete(dists, i)
            dists = np.sort(dists)
            
            # Check for 0 distance (repeated points)
            if dists[0] < 0.05:
                return 0.0
                
            # Find min gap
            diffs = np.diff(dists)
            k = np.argmin(diffs)
            D = (dists[k] + dists[k+1]) / 2.0
            if D < 0.1:
                return 0.0
                
            # Select 4 distances
            if k == 0:
                sel = dists[0:4]
            elif k == 17: # n-2 = 18? No, indices 0..18. k max is 17.
                sel = dists[15:19]
            else:
                sel = dists[k-1:k+3]
                
            ratios = np.maximum(sel / D, D / sel)
            raw = np.mean(ratios)
            score = min(1.0, 1.0 / raw)
            min_vertex_score = min(min_vertex_score, score)
            
        return min_vertex_score

    # Optimization
    # We want to maximize score, so minimize -score
    # Use a simple local search or scipy
    
    # To ensure speed, we'll do a few steps of gradient-free search
    best_score = calc_score(radii)
    best_radii = radii.copy()
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(100):
        perturbation = np.random.normal(0, 0.05, n)
        new_radii = best_radii + perturbation
        # Ensure radii are positive
        new_radii = np.maximum(new_radii, 0.5)
        s = calc_score(new_radii)
        if s > best_score:
            best_score = s
            best_radii = new_radii
    
    # Final convexity check
    pts = np.column_stack([best_radii * np.cos(angles), best_radii * np.sin(angles)])
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    dists_to_centroid = np.linalg.norm(pts, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    pts = pts / mean_dist
    
    # Check strict convexity
    # Calculate cross products of consecutive edges
    cross_products = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1)%n]
        p2 = pts[(i+2)%n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_products.append(cross)
        
    if np.any(np.sign(cross_products) != np.sign(cross_products[0])) or np.any(np.abs(cross_products) < 1e-6):
        # Fallback to regular polygon if invalid
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return pts
