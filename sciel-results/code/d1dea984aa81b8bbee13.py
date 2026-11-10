# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _triangle_areas(points):
    """
    Compute the areas of all triangles formed by triples of points.
    Returns an array of shape (286,) for n=13.
    """
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # 2 * Area = |det(p2-p1, p3-p1)|
                det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas[i, j, k] = 0.5 * np.sqrt(det*det + 1e-12)
    return areas

def _soft_min_area(points, beta, eps=1e-12):
    """
    Compute a smooth approximation of the minimum triangle area.
    We want to maximize this, so we will minimize its negative.
    """
    n = len(points)
    areas = np.zeros((n, n, n))
    # Vectorized calculation for speed? 
    # For n=13, loops are fine.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                areas[i, j, k] = 0.5 * np.sqrt(det*det + eps)
    
    # Flatten to get all 286 areas
    # Only upper triangle part is filled
    idx = np.triu_indices(n, k=2)
    areas_flat = areas[idx]
    
    # Log-sum-exp for stability
    # -1/beta * log(sum(exp(-beta * area)))
    # This approximates -min(area) as beta -> inf
    # So we want to MINIMIZE this value? 
    # If min_area is large, exp(-beta*large) is small, sum is small, log is negative large, -1/beta * neg_large is positive large.
    # Wait.
    # Let f(x) = 1/beta * log(sum(exp(-beta * x_i))).
    # If x_i are all equal to A, f(A) = 1/beta * log(N * exp(-beta*A)) = 1/beta * (log N - beta A) = (log N)/beta - A.
    # So f(A) approx -A.
    # We want to maximize A. So we want to minimize -A.
    # So minimizing f(x) is correct.
    
    max_val = np.max(-beta * areas_flat)
    sum_exp = np.sum(np.exp(-beta * areas_flat - max_val))
    val = (max_val + np.log(sum_exp)) / beta
    return val

def _objective(params, beta, hull_points):
    """
    params: flattened coordinates of the 7 interior points (14 values).
    Returns the value to minimize (negative soft-min area).
    """
    # Reshape params to (7, 2)
    interior_points = params.reshape(-1, 2)
    
    # Combine with hull points
    all_points = np.vstack((hull_points, interior_points))
    
    # Calculate soft min area
    score = _soft_min_area(all_points, beta)
    
    # We want to maximize the area, so we minimize -score
    # But _soft_min_area returns approx -min_area.
    # So if min_area is high, _soft_min_area is very negative.
    # We want to maximize min_area -> minimize -min_area.
    # So we should minimize - _soft_min_area?
    # Let's check:
    # If min_area = 10, soft_min ~ -10.
    # We want to maximize 10.
    # So we want to minimize -10? No, we want to go from -10 to -20 (which means area 20).
    # So minimizing the returned value is correct.
    return score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    """
    start_time = time.time()
    deadline = start_time + 600.0 # Safety margin
    
    # 1. Define the Hull (Regular Hexagon)
    # Vertices of a regular hexagon with circumradius 1
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hull_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # 2. Initial Guess for Interior Points
    # Place 7 points: Center + 6 centroids of the triangles formed by center and adjacent hull vertices
    center = np.array([0.0, 0.0])
    interior_init = []
    for i in range(6):
        v1 = hull_points[i]
        v2 = hull_points[(i+1)%6]
        centroid = (center + v1 + v2) / 3.0
        interior_init.append(centroid)
    interior_init.append(center)
    interior_init = np.array(interior_init)
    
    # Flatten initial parameters
    x0 = interior_init.flatten()
    
    best_points = np.vstack((hull_points, interior_init))
    best_score = -np.inf # We are minimizing the objective (which is ~ -area)
    
    # 3. Optimization Loop
    # Use a sequence of beta values (annealing)
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline - 60.0:
            break
            
        # Define the objective function for this beta
        def obj(p):
            return _objective(p, beta, hull_points)
        
        # Gradient can be computed numerically by L-BFGS-B
        # Or we can provide it. Numerical is safer for the sqrt term.
        res = minimize(
            obj, 
            current_x, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
        )
        
        current_x = res.x
        current_points = np.vstack((hull_points, current_x.reshape(-1, 2)))
        
        # Calculate actual score for tracking
        # We need the real min area / hull area
        # Hull area is fixed for regular hexagon R=1: 3*sqrt(3)/2
        hull_area = 3 * np.sqrt(3) / 2
        areas = _triangle_areas(current_points)
        idx = np.triu_indices(13, k=2)
        min_area = np.min(areas[idx])
        score = min_area / hull_area
        
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    # Ensure the result is valid (finite, correct shape)
    if not np.all(np.isfinite(best_points)):
        # Fallback to initial
        best_points = np.vstack((hull_points, interior_init))
        
    return best_points

# EVOLVE-BLOCK-END
