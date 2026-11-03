import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

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
    
    pts[:, 0] = np.clip(pts[:, 0], eps, 1 - eps)
    pts[:, 1] = np.clip(pts[:, 1], eps, np.sqrt(3)/2 - eps)
    
    for i in range(len(pts)):
        x, y = pts[i]
        val_right = np.sqrt(3)*x + y - np.sqrt(3)
        if val_right > 0:
            pts[i, 0] = x - (val_right / 4.0) * np.sqrt(3)
            pts[i, 1] = y - (val_right / 4.0)
        
        val_left = -np.sqrt(3)*x + y
        if val_left > 0:
            pts[i, 0] = x - (val_left / 4.0) * (-np.sqrt(3))
            pts[i, 1] = y - (val_left / 4.0)
            
        pts[i, 0] = np.clip(pts[i, 0], eps, 1 - eps)
        pts[i, 1] = np.clip(pts[i, 1], eps, np.sqrt(3)/2 - eps)
        
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def objective(x):
        """
        Objective function for minimization.
        x: flat array of shape (2*N,)
        Returns: -min_area
        """
        pts = x.reshape((N, 2))
        pts = _project_to_triangle(pts)
        return -_min_triangle_area_vec(pts)
    
    # Bounds for each coordinate
    # x in [0, 1], y in [0, sqrt(3)/2]
    # We use slightly tighter bounds to help the optimizer stay in the feasible region
    # though projection handles the rest.
    x_min = 1e-4
    x_max = 1 - 1e-4
    y_min = 1e-4
    y_max = np.sqrt(3)/2 - 1e-4
    
    bounds = [(x_min, x_max), (y_min, y_max)] * N
    
    best_points = None
    best_area = 0.0
    
    # Phase 1: Differential Evolution
    # DE is robust for global optimization.
    # popsize: 15 (330 individuals)
    # maxiter: 100 (33000 evaluations)
    # tol: 1e-6
    # strategy: 'best1bin' is standard and effective.
    
    try:
        # We need to be careful with time. DE can be slow if it runs to completion.
        # We'll let it run but check time in a callback if possible, or just set conservative limits.
        # scipy's differential_evolution does not have a direct time-based callback that stops it cleanly without exceptions in older versions,
        # but we can use 'callback' to check time and raise an exception or just rely on maxiter.
        # Given 150s limit, and ~165 triples per eval, we can afford ~10k-20k evals.
        # Let's aim for ~10k evals in DE.
        # popsize=10, maxiter=1000 -> 10000 evals.
        
        res_de = differential_evolution(
            objective, 
            bounds, 
            popsize=10, 
            maxiter=1000, 
            tol=1e-6, 
            seed=42,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        
        if res_de.success or (res_de.x is not None):
            candidate_pts = _project_to_triangle(res_de.x.reshape((N, 2)))
            candidate_area = _min_triangle_area_vec(candidate_pts)
            if candidate_area > best_area:
                best_area = candidate_area
                best_points = candidate_pts
    except Exception as e:
        print(f"DE failed: {e}")

    # If DE didn't find anything or we have time, try a few random restarts with local search
    # to ensure we have a good starting point if DE got stuck or was too slow.
    # However, DE usually finds a decent global candidate.
    
    # Phase 2: Local Polish with Nelder-Mead
    # Use the best point from DE as the start.
    if best_points is None:
        # Fallback if DE failed completely
        rng = np.random.default_rng(42)
        best_points = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * (np.sqrt(3)/2)
                if y <= np.sqrt(3)*(1-x) - 1e-9 and y >= 1e-9 and x >= 1e-9 and x <= 1-1e-9:
                    best_points[i] = [x, y]
                    break
        best_points = _project_to_triangle(best_points)
        best_area = _min_triangle_area_vec(best_points)
        
    if time.time() < deadline - 5.0:
        # Nelder-Mead is very good for smooth local optimization.
        # We want to maximize min_area, so minimize -min_area.
        
        # We might run NM multiple times or with higher precision.
        # NM can be sensitive to initial simplex size.
        
        try:
            # Increase maxiter for better convergence
            # xatol and fatol for precision
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            
            if res_nm.x is not None:
                refined_pts = _project_to_triangle(res_nm.x.reshape((N, 2)))
                refined_area = _min_triangle_area_vec(refined_pts)
                if refined_area > best_area:
                    best_area = refined_area
                    best_points = refined_pts
        except Exception as e:
            print(f"NM failed: {e}")

    # If we still have time, try a second round of DE or NM from a different start?
    # Or maybe a "shuffled" NM?
    # Given the time budget, one good DE + one good NM is a strong strategy.
    # Let's check if we can afford another NM polish with different initial simplex or a second DE.
    
    if time.time() < deadline - 10.0 and best_points is not None:
        # Try to perturb the best solution slightly and polish again?
        # Or run DE again with a different seed?
        # DE with different seed might find a different local minimum which could be better.
        # But DE is expensive.
        # Let's try a "multi-start" NM from the DE result and a random point.
        
        rng = np.random.default_rng(123)
        
        # Random start
        rand_pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                x = rng.random()
                y = rng.random() * (np.sqrt(3)/2)
                if y <= np.sqrt(3)*(1-x) - 1e-9 and y >= 1e-9 and x >= 1e-9 and x <= 1-1e-9:
                    rand_pts[i] = [x, y]
                    break
        rand_pts = _project_to_triangle(rand_pts)
        
        # Polish random start
        try:
            res_nm2 = minimize(
                objective, 
                rand_pts.flatten(), 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            if res_nm2.x is not None:
                refined_pts2 = _project_to_triangle(res_nm2.x.reshape((N, 2)))
                refined_area2 = _min_triangle_area_vec(refined_pts2)
                if refined_area2 > best_area:
                    best_area = refined_area2
                    best_points = refined_pts2
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
