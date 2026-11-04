import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

N = 13

# EVOLVE-BLOCK-START
def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    Returns negative score for minimization.
    """
    points = np.asarray(points)
    
    # 1. Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0 # Degenerate hull
    except Exception:
        return 0.0

    # 2. Calculate Minimum Triangle Area
    min_area = float('inf')
    n = len(points)
    
    # Vectorized calculation for all triples
    # Create indices for all combinations of 3
    # Instead of itertools in a loop, we can use broadcasting for speed, 
    # but 286 is small. Let's stick to a slightly optimized loop or vectorization.
    # Vectorization:
    # i, j, k indices
    # This might be memory heavy if done naively, but 286 is tiny.
    # Let's just use the loop, it's fast enough for 26 params.
    for i, j, k in itertools.combinations(range(n), 3):
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        if area < min_area:
            min_area = area
            
    if min_area <= 1e-12:
        return 0.0
        
    score = min_area / hull_area
    return -score

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for safety
    
    # Initial best candidate: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    best_points = np.column_stack((x, y))
    best_score = -_score(best_points)
    
    # Multi-start Nelder-Mead
    # We will run several iterations with different random initializations
    # and also perturbations of the regular polygon.
    
    num_starts = 20
    max_iter_per_start = 500
    
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Strategy 1: Random points in unit circle
        # Strategy 2: Perturbed regular polygon
        
        if i % 2 == 0:
            # Random points
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.5, 1.0, N) # Avoid center to ensure convexity potential
            x = radii * np.cos(angles)
            y = radii * np.sin(angles)
        else:
            # Perturbed regular polygon
            base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            # Add small noise to angles and radii
            angle_noise = rng.normal(0, 0.1, N)
            radius_noise = rng.normal(1.0, 0.1, N)
            x = radius_noise * np.cos(base_angles + angle_noise)
            y = radius_noise * np.sin(base_angles + angle_noise)
            
        x0 = np.column_stack((x, y)).flatten()
        
        # Run Nelder-Mead
        try:
            res = minimize(_score, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter_per_start, 'xatol': 1e-4, 'fatol': 1e-6})
            opt_points = res.x.reshape(-1, 2)
            current_score = -_score(opt_points)
            
            if current_score > best_score:
                best_score = current_score
                best_points = opt_points
        except Exception:
            continue
            
    # Final polish with the best points found so far
    # Increase iteration count for the final refinement
    if time.time() - start_time < time_limit - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(_score, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            opt_points = res.x.reshape(-1, 2)
            final_score = -_score(opt_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = opt_points
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
