import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# Pre-compute triangle indices for efficiency
# There are 286 triangles
TRI_INDICES = list(combinations(range(N), 3))

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute areas of all triangles
    # Vectorized calculation for speed
    # points shape (N, 2)
    # tri_indices shape (286, 3)
    p1 = points[TRI_INDICES[:, 0]]
    p2 = points[TRI_INDICES[:, 1]]
    p3 = points[TRI_INDICES[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_tri_area = np.min(areas)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        # If points are collinear or degenerate, hull area is 0
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    return min_tri_area / hull_area

def _objective(params):
    """
    Objective function for differential evolution.
    Returns negative score to be minimized.
    """
    R0 = params[0]
    # K=3 Gaussians
    # params: R0, w1, mu1, w2, mu2, w3, mu3
    # Fixed sigma = 1.0 radian for smoothness
    
    K = 3
    sigma = 1.0
    
    # Sample 13 angles uniformly
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Compute radius for each angle
    radii = np.full(N, R0)
    for k in range(K):
        w = params[1 + 2*k]
        mu = params[2 + 2*k]
        # Periodic distance for angle
        diff = thetas - mu
        # Wrap difference to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        radii += w * np.exp(-(diff**2) / (2 * sigma**2))
        
    # Ensure radii are positive to avoid singularities/weird geometry
    radii = np.maximum(radii, 0.1)
    
    # Convert to cartesian
    points = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
    
    score = _compute_score(points)
    
    # Penalize if shape is not "nice" (e.g. very small hull area relative to max radius)
    # But score already normalizes by hull area.
    # If hull area is 0, score is 0.
    
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin for safety
    
    # Initial best
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    thetas_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(thetas_init), np.sin(thetas_init)))
    best_points = points_init
    best_score = _compute_score(points_init)
    
    # Define bounds for Differential Evolution
    # R0 in [0.5, 1.5]
    # w in [-0.5, 0.5]
    # mu in [0, 2*pi]
    bounds = [
        (0.5, 1.5),
        (-0.5, 0.5), (0, 2 * np.pi),
        (-0.5, 0.5), (0, 2 * np.pi),
        (-0.5, 0.5), (0, 2 * np.pi)
    ]
    
    # Run DE
    try:
        # maxiter is a soft limit, we check time in callback
        def time_check(res):
            if time.time() > deadline:
                # This doesn't stop DE directly in all versions, 
                # but we will check time after DE or use a wrapper.
                # Actually, DE in scipy doesn't have a built-in time limit callback that stops it cleanly in all versions.
                # We will rely on maxiter being conservative or just let it run and check time after.
                # Better approach: Run DE with a reasonable maxiter and polish.
                pass

        # To be safe with time, we run DE with a limited number of iterations
        # and then do a local polish if time permits.
        # 7 parameters, DE is fast.
        
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-6,
            popsize=15
        )
        
        if time.time() < deadline:
            # Check result
            params_opt = result.x
            R0 = params_opt[0]
            K = 3
            sigma = 1.0
            thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.full(N, R0)
            for k in range(K):
                w = params_opt[1 + 2*k]
                mu = params_opt[2 + 2*k]
                diff = thetas - mu
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                radii += w * np.exp(-(diff**2) / (2 * sigma**2))
            radii = np.maximum(radii, 0.1)
            points_opt = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
            
            score_opt = _compute_score(points_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_points = points_opt

        # If time remains, try a local refinement using Nelder-Mead on the parameters
        if time.time() < deadline - 5.0 and best_points is not None:
            # We need to refine the points directly or the params?
            # Refining params is safer to maintain structure.
            # Let's try refining the 7 params with Nelder-Mead
            from scipy.optimize import minimize
            
            # Use the best params found by DE as starting point
            x0 = result.x if result.success else [1.0, 0, 0, 0, 0, 0, 0]
            
            res_nm = minimize(
                _objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if time.time() < deadline:
                params_nm = res_nm.x
                R0 = params_nm[0]
                K = 3
                sigma = 1.0
                thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
                radii = np.full(N, R0)
                for k in range(K):
                    w = params_nm[1 + 2*k]
                    mu = params_nm[2 + 2*k]
                    diff = thetas - mu
                    diff = (diff + np.pi) % (2 * np.pi) - np.pi
                    radii += w * np.exp(-(diff**2) / (2 * sigma**2))
                radii = np.maximum(radii, 0.1)
                points_nm = np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas)))
                
                score_nm = _compute_score(points_nm)
                if score_nm > best_score:
                    best_score = score_nm
                    best_points = points_nm

    except Exception as e:
        # If optimization fails, return initial points
        pass
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To match the required interface strictly, we ensure construct is the entry point.
# The code above defines construct.
# EVOLVE-BLOCK-END
