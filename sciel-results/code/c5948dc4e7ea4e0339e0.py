import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _objective(x):
    # x is flattened array of shape (2*N,)
    points = x.reshape((N, 2))
    # We want to maximize score, so minimize negative score
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.array([np.cos(angles), np.sin(angles)]).T.flatten()
    
    best_x = x0
    best_score = _score(x0.reshape((N, 2)))
    
    # Try local optimization
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < -best_score: # i.e. -score < -best_score => score > best_score
            best_x = res.x
            best_score = -res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        return best_x.reshape((N, 2))
        
    # If we have time, try a few other initializations or restarts
    # For now, one good run from regular polygon is often sufficient to beat random
    # But let's try a second run from a slightly perturbed regular polygon or a grid if time permits.
    
    # Actually, Nelder-Mead might get stuck. Let's try Powell as well if time allows.
    if time.time() < deadline - 30.0:
        try:
            res2 = minimize(_objective, x0, method='Powell', 
                            options={'maxiter': 200, 'ftol': 1e-8})
            if res2.fun < -best_score:
                best_x = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    return best_x.reshape((N, 2))
