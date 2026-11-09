import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius of a set of points on the unit sphere.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    # Compute Convex Hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, return a large value (e.g., 180 degrees)
        return np.pi
    
    max_angle = 0.0
    
    # Iterate over simplices (triangles)
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Calculate circumcenter of the triangle on the sphere
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        
        numerator = cross_bc + cross_ca + cross_ab
        denominator = 2.0 * np.dot(a, np.cross(b, c))
        
        if np.abs(denominator) < 1e-10:
            # Degenerate
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            local_max = max(d_ab, d_bc, d_ca)
            if local_max > max_angle:
                max_angle = local_max
            continue
            
        circumcenter = numerator / denominator
        
        # Normalize circumcenter
        norm_cc = np.linalg.norm(circumcenter)
        if norm_cc < 1e-10:
            continue
        circumcenter = circumcenter / norm_cc
        
        # Calculate angle
        angle = np.arccos(np.clip(np.dot(circumcenter, a), -1.0, 1.0))
        
        if angle > max_angle:
            max_angle = angle
            
    return max_angle

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # Initial configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = init_points.copy()
    best_radius = _covering_radius(best_points)
    
    def objective(params):
        points = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        return _covering_radius(points)
    
    # Optimization
    restarts = 0
    max_restarts = 10
    
    while time.time() - start_time < timeout:
        # If we have time, try a new restart
        if restarts > 0:
            # Perturb the best points found so far
            noise = np.random.normal(0, 0.2, (N, 3))
            init_points = best_points + noise
            init_points = init_points / np.linalg.norm(init_points, axis=1, keepdims=True)
        
        # Run optimizer
        try:
            res = minimize(objective, init_points.flatten(), method='Powell', 
                           options={'maxiter': 2000, 'maxfev': 10000})
            
            if res.success or res.fun < best_radius:
                new_points = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                norms[norms < 1e-10] = 1.0
                new_points = new_points / norms
                new_radius = _covering_radius(new_points)
                
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points
        except Exception:
            pass
            
        restarts += 1
        
        # Check time
        if time.time() - start_time > timeout - 5.0:
            break
            
    return best_points
