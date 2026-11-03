import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2

def _min_triangle_area_vec(points):
    """
    Vectorized calculation of the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    Returns: float, the minimum area.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            p3s = points[j+1:]
            v13s = p3s - p1
            cross_vals = v12[0] * v13s[:, 1] - v12[1] * v13s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            min_local = np.min(areas)
            if min_local < min_area:
                min_area = min_local
    return min_area

def _project_to_triangle(points):
    """
    Project points onto the interior of the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    eps = 1e-9
    pts = np.array(points, dtype=float)
    
    # Clip to bounding box first
    pts[:, 0] = np.clip(pts[:, 0], eps, 1 - eps)
    pts[:, 1] = np.clip(pts[:, 1], eps, H - eps)
    
    for i in range(len(pts)):
        x, y = pts[i]
        # Check right edge: sqrt(3)*x + y - sqrt(3) <= 0
        val_right = np.sqrt(3)*x + y - np.sqrt(3)
        if val_right > 0:
            pts[i, 0] = x - (val_right / 4.0) * np.sqrt(3)
            pts[i, 1] = y - (val_right / 4.0)
        
        # Check left edge: -sqrt(3)*x + y <= 0
        val_left = -np.sqrt(3)*x + y
        if val_left > 0:
            pts[i, 0] = x - (val_left / 4.0) * (-np.sqrt(3))
            pts[i, 1] = y - (val_left / 4.0)
            
        pts[i, 0] = np.clip(pts[i, 0], eps, 1 - eps)
        pts[i, 1] = np.clip(pts[i, 1], eps, H - eps)
        
    return pts

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # Define bounds for each coordinate
    # x in [0, 1], y in [0, H]
    # The projection handles the triangle constraint, so we can use box bounds for DE
    bounds = [(0.0, 1.0)] * N + [(0.0, H)] * N
    
    # Objective function for DE (minimize negative area)
    def objective(x):
        pts = x.reshape((N, 2))
        pts = _project_to_triangle(pts)
        area = _min_triangle_area_vec(pts)
        return -area

    best_points = None
    best_area = 0.0
    
    # Use Differential Evolution
    # maxiter and popsize are tuned to fit within time limit
    # DE is stochastic, so we use a fixed seed for reproducibility if needed, 
    # but the task allows randomness. We'll use a seed for consistency in evaluation.
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-4,
            polish=False, # We will do our own polish to control time
            init='sobol' # Better coverage of the space
        )
        
        # res.x is the best point found
        candidate_pts = _project_to_triangle(res.x.reshape((N, 2)))
        candidate_area = _min_triangle_area_vec(candidate_pts)
        
        if candidate_area > best_area:
            best_area = candidate_area
            best_points = candidate_pts
            
    except Exception as e:
        # Fallback if DE fails or times out unexpectedly
        pass

    # If DE didn't run or we have time, do a few local restarts from the DE result
    # or from random if DE failed
    if best_points is None:
        # Fallback random init
        rng = np.random.default_rng(42)
        best_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * H
                if y <= np.sqrt(3)*(1-x) and y >= 0 and x >= 0 and x <= 1:
                    best_points[i] = [x, y]
                    break
        best_points = _project_to_triangle(best_points)
        best_area = _min_triangle_area_vec(best_points)

    # Final Polish with Nelder-Mead
    # This is crucial for squeezing out the last bits of area
    if time.time() < deadline - 5.0:
        def obj_local(p):
            pts = _project_to_triangle(p.reshape((N, 2)))
            return -_min_triangle_area_vec(pts)
        
        try:
            # Start from the DE result
            res_nm = minimize(
                obj_local, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            refined_pts = _project_to_triangle(res_nm.x.reshape((N, 2)))
            refined_area = _min_triangle_area_vec(refined_pts)
            
            if refined_area > best_area:
                best_area = refined_area
                best_points = refined_pts
                
        except Exception:
            pass

    # If we have significant time left, we can try a second DE run with different seed
    # or a second Nelder-Mead from a slightly perturbed start
    if time.time() < deadline - 10.0 and best_points is not None:
        # Try a local refinement from a perturbed version of the best point
        rng = np.random.default_rng(123)
        perturbed_pts = best_points + rng.normal(0, 0.01, size=(N, 2))
        perturbed_pts = _project_to_triangle(perturbed_pts)
        
        def obj_local2(p):
            pts = _project_to_triangle(p.reshape((N, 2)))
            return -_min_triangle_area_vec(pts)
            
        try:
            res_nm2 = minimize(
                obj_local2, 
                perturbed_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            refined_pts2 = _project_to_triangle(res_nm2.x.reshape((N, 2)))
            refined_area2 = _min_triangle_area_vec(refined_pts2)
            
            if refined_area2 > best_area:
                best_area = refined_area2
                best_points = refined_pts2
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
