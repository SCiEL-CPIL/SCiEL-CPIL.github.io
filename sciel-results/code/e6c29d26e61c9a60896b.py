import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def get_score(points):
    """
    Computes the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except:
        return 0.0

    # Vectorized triangle area computation
    # Generate all C(N,3) index triples
    # For N=13, C(13,3) = 286, which is small enough to handle easily
    idx = np.array(list(itertools.combinations(range(N), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Cross product magnitude gives 2 * area
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def soft_min(x, k):
    """
    Smooth approximation of the minimum.
    As k -> infinity, it approaches min(x).
    """
    return -1.0 / k * np.log(np.sum(np.exp(-k * x)))

def objective(params, k):
    """
    Objective function to minimize: -score using a soft-min surrogate.
    params: flattened array of 2D coordinates for 13 points (26 values).
    """
    points = params.reshape(13, 2)
    
    # Basic sanity check: points should be distinct
    if np.any(np.diff(np.sort(points.flatten())) < 1e-8):
        return 1e10

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e10
    except:
        return 1e10

    # Calculate all triangle areas
    idx = np.array(list(itertools.combinations(range(13), 3)))
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Use soft-min to smooth the objective
    min_area_smooth = soft_min(areas, k)
    
    score = min_area_smooth / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = get_score(points_init)
    best_points = points_init.copy()
    
    # Parameters for optimization
    # We will optimize the raw coordinates directly to allow hull shape variation.
    # This is a 26-dimensional space.
    
    # We use Powell's method because it doesn't require gradients (which are noisy
    # due to the min function, even if smoothed) and handles non-linear constraints
    # implicitly well through the objective function penalties.
    
    # We will do multiple restarts with different initial configurations and
    # different 'k' values for the soft-min to balance exploration vs exploitation.
    
    n_restarts = 0
    max_restarts = 50 # Cap restarts, but loop until time runs out
    
    # List of k values to try. Higher k is closer to true min (harder to optimize),
    # lower k is smoother (easier to optimize but might find suboptimal minima).
    k_values = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    while time.time() < deadline and n_restarts < max_restarts:
        n_restarts += 1
        
        # Choose k: Start with lower k for exploration, higher for exploitation
        # Simple strategy: cycle through k values
        k = k_values[n_restarts % len(k_values)]
        
        # Generate initial guess
        if n_restarts == 1:
            x0 = best_points.flatten()
        else:
            # Perturb the best found solution or start from a random regular polygon
            if n_restarts % 3 == 0:
                # Random regular polygon with random rotation and scale
                theta = np.random.uniform(0, 2 * np.pi)
                scale = np.random.uniform(0.5, 1.5)
                angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + theta
                pts = np.column_stack((scale * np.cos(angles), scale * np.sin(angles)))
                x0 = pts.flatten()
            else:
                # Perturb best points
                noise_scale = 0.1 + 0.2 * np.random.rand()
                x0 = best_points.flatten() + np.random.normal(0, noise_scale, 2 * N)
        
        # Run Powell optimization
        try:
            res = minimize(objective, x0, args=(k,), method='Powell',
                           options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-8})
            
            if res.fun < -1e9: # Check for invalid penalty
                continue
                
            current_score = -res.fun
            
            # If the score improved, update best
            if current_score > best_score:
                # Verify with exact min (not soft-min) to ensure it's a valid improvement
                pts_test = res.x.reshape(N, 2)
                exact_score = get_score(pts_test)
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = pts_test.copy()
        except Exception:
            pass
        
        # Check time frequently
        if time.time() > deadline - 5:
            break

    # Final check: If we have time, do a local refinement of the best point using
    # a very high k (almost exact min) with Nelder-Mead for fine-tuning
    if time.time() < deadline - 10 and best_points is not None:
        try:
            k_final = 1000.0
            x_final = best_points.flatten()
            res_final = minimize(objective, x_final, args=(k_final,), method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if res_final.fun < -1e9:
                return best_points
            
            final_score = -res_final.fun
            pts_final = res_final.x.reshape(N, 2)
            exact_final_score = get_score(pts_final)
            
            if exact_final_score > best_score:
                best_score = exact_final_score
                best_points = pts_final.copy()
        except Exception:
            pass

    return best_points

# Ensure the function signature matches the requirement
# The task says: construct() -> array of shape (n, 2)
# My construct() returns best_points which is (13, 2)
