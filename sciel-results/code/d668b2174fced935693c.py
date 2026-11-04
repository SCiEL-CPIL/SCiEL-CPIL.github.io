import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

N = 14

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.array(points)
    n = len(points)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        # Fallback if hull fails (e.g., collinear points)
        # Simple shoelace on sorted by angle? Or just return 0
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    # Iterate over all triples
    # For N=14, there are 364 triples. This is fast.
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-9:
                        return 0.0

    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Regular 14-gon
    x0 = _regular_polygon(N)
    x0 = x0.flatten()
    
    current_best_score = _score(x0.reshape(-1, 2))
    best_points = x0.reshape(-1, 2).copy()
    
    # Objective function to minimize (negative score)
    def objective(x):
        points = x.reshape(-1, 2)
        return -_score(points)
    
    # Try Nelder-Mead
    # This is a local search. It might get stuck in local minima.
    # We can try a few restarts with slight perturbations if time permits.
    
    try:
        # Nelder-Mead parameters
        # initial_simplex_size is important
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if time.time() < deadline:
            score_res = -res.fun
            if score_res > current_best_score:
                current_best_score = score_res
                best_points = res.x.reshape(-1, 2)
                
    except Exception as e:
        pass

    # If time remains, try a few random restarts around the best solution
    # or simple local perturbations
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb the best solution
            noise_scale = 0.05
            x_perturbed = best_points.flatten() + np.random.normal(0, noise_scale, size=2*N)
            
            try:
                res2 = minimize(objective, x_perturbed, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if time.time() > deadline:
                    break
                    
                score_res2 = -res2.fun
                if score_res2 > current_best_score:
                    current_best_score = score_res2
                    best_points = res2.x.reshape(-1, 2)
            except Exception:
                pass

    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    points = np.array(points)
    n = len(points)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                v1 = pj - pi
                v2 = pk - pi
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-9:
                        return 0.0

    return min_area / hull_area

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Regular 14-gon
    x0 = _regular_polygon(N)
    x0 = x0.flatten()
    
    current_best_score = _score(x0.reshape(-1, 2))
    best_points = x0.reshape(-1, 2).copy()
    
    # Objective function to minimize (negative score)
    def objective(x):
        points = x.reshape(-1, 2)
        return -_score(points)
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if time.time() < deadline:
            score_res = -res.fun
            if score_res > current_best_score:
                current_best_score = score_res
                best_points = res.x.reshape(-1, 2)
                
    except Exception:
        pass

    # If time remains, try a few random restarts around the best solution
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb the best solution
            noise_scale = 0.05
            x_perturbed = best_points.flatten() + np.random.normal(0, noise_scale, size=2*N)
            
            try:
                res2 = minimize(objective, x_perturbed, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
                
                if time.time() > deadline:
                    break
                    
                score_res2 = -res2.fun
                if score_res2 > current_best_score:
                    current_best_score = score_res2
                    best_points = res2.x.reshape(-1, 2)
            except Exception:
                pass

    return best_points
# EVOLVE-BLOCK-END
