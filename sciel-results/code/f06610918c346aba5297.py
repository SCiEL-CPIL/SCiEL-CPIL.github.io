"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed where possible, but N=14 is small enough for simple loops.
    """
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    # Iterate over all pairs (i, j) and compute cross product with all k > j
    # This avoids redundant calculations and keeps memory low.
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # Vectors from i to all k > j
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            # Cross product magnitude gives 2 * area
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    """
    Objective function for minimization.
    We want to maximize score, so we minimize -score.
    """
    points = points_flat.reshape(N, 2)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    
    Strategy:
    1. Initialize with a regular 14-gon. This is a strong starting point because
       the optimal configuration for small N is often close to a regular polygon
       or a slight perturbation thereof to break symmetry and improve the min-area.
    2. Use Nelder-Mead for robust local refinement. It does not require gradients
       and handles the non-smoothness of the 'min' operation reasonably well.
    3. Use Powell's method for a final polish, which is very effective for
       smooth landscapes and can squeeze out more precision.
    4. Normalize the result to unit area.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # 1. Initialization: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Scale to unit area initially? No, scale doesn't matter for the ratio, 
    # but it helps numerical stability to keep coordinates around O(1).
    # Area of regular n-gon with circumradius R is 0.5 * n * R^2 * sin(2*pi/n).
    # We can just use R=1.
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = reg_points.flatten()
    
    best_result = x0.copy()
    best_score_val = _score(best_result.reshape(N, 2))
    
    # 2. Nelder-Mead Refinement
    if time.time() < deadline - 30:
        try:
            res_nm = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True,
                    'initial_simplex': None # Let it generate, or we could provide one
                }
            )
            current_score = -res_nm.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res_nm.x
        except Exception:
            pass
    
    # 3. Powell's Method Polish
    if time.time() < deadline - 10:
        try:
            res_powell = minimize(
                _objective, 
                best_result, 
                method='Powell', 
                options={
                    'maxiter': 1000, 
                    'xtol': 1e-10, 
                    'ftol': 1e-12
                }
            )
            current_score = -res_powell.fun
            if current_score > best_score_val:
                best_score_val = current_score
                best_result = res_powell.x
        except Exception:
            pass
            
    # 4. Final Verification and Normalization
    final_points = best_result.reshape(N, 2)
    
    try:
        hull = ConvexHull(final_points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        final_points = reg_points.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
    else:
        # Degenerate case, return regular polygon normalized
        final_points = reg_points.copy()
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
# EVOLVE-BLOCK-END
