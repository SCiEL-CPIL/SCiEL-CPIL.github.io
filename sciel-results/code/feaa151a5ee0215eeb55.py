import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _triangle_area(p1, p2, p3):
    """Calculate area of triangle given 3 points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Find the minimum area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Calculate the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _neg_score(points_flat):
    """Objective function for minimization. Input is flat array of 2N floats."""
    points = points_flat.reshape(N, 2)
    score = _score(points)
    # We want to maximize score, so minimize -score
    # Add a small penalty if points are not in convex position to guide optimization
    # Check convexity: all cross products of consecutive edges (in CCW order) should have same sign
    # Sort points by angle around centroid to ensure consistent ordering
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    convex_penalty = 0.0
    n = N
    for i in range(n):
        p0 = sorted_points[i]
        p1 = sorted_points[(i + 1) % n]
        p2 = sorted_points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross < 0:
            convex_penalty += abs(cross)
            
    return -score + 100.0 * convex_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Try a few different initializations and optimizations
    # We will use Powell method which is robust for non-smooth functions
    
    num_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a perturbed regular polygon
        scale = 1.0
        rotation = rng.uniform(0, 2 * np.pi)
        x = np.cos(angles + rotation) * scale
        y = np.sin(angles + rotation) * scale
        
        # Add small random perturbations
        x += rng.normal(0, 0.05, N)
        y += rng.normal(0, 0.05, N)
        
        initial_flat = np.column_stack([x, y]).flatten()
        
        # Optimize
        try:
            res = minimize(
                _neg_score, 
                initial_flat, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6}
            )
            
            optimized_points = res.x.reshape(N, 2)
            current_score = _score(optimized_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = optimized_points.copy()
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to regular polygon if optimization failed
        best_points = init_points
        best_score = _score(best_points)
        
    # Normalize to unit area? The problem says "normalized to unit area" but the score is 
    # "minimum triangle area divided by the convex hull area". 
    # This ratio is scale-invariant. So we don't need to explicitly normalize the coordinates 
    # for the score calculation, but the output should be a valid set of points.
    # The interface expects an array of shape (n, 2).
    
    return best_points
