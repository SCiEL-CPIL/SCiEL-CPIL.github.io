import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    n = 8
    
    def get_points(r1, s):
        # r1: distance from origin to axis points
        # s: coordinate of diagonal points (x=x, y=s)
        p1 = np.array([r1, 0])
        p2 = np.array([-r1, 0])
        p3 = np.array([0, r1])
        p4 = np.array([0, -r1])
        p5 = np.array([s, s])
        p6 = np.array([-s, s])
        p7 = np.array([-s, -s])
        p8 = np.array([s, -s])
        return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

    def objective(params):
        r1, s = params
        if r1 < 1e-6 or s < 1e-6:
            return 1e10
        points = get_points(r1, s)
        
        # Calculate all pairwise distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-10:
                    return 1e10
                dists.append(d)
        
        dists = np.array(dists)
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        val = np.sum(np.log(dists)) - 28 * np.log(D)
        return -val

    # Initial guesses
    # Regular octagon: R=1, r1=1, s=1/sqrt(2)
    p_reg = [1.0, 1.0 / np.sqrt(2)]
    
    # Try other ratios
    # If s is larger, the diagonal points move out.
    # If r1 is larger, axis points move out.
    
    starts = [
        [1.0, 1.0 / np.sqrt(2)], # Regular octagon
        [1.0, 0.8],
        [1.0, 0.6],
        [1.2, 1.0 / np.sqrt(2)],
        [0.8, 1.0 / np.sqrt(2)],
        [1.0, 1.0], # Square + Square (one rotated 45 deg, same radius) -> Diamond + Cross
        [1.5, 0.5],
        [0.5, 1.5]
    ]
    
    best_val = -np.inf
    best_params = None
    
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback to regular octagon
        r1, s = 1.0, 1.0 / np.sqrt(2)
    else:
        r1, s = best_params

    points = get_points(r1, s)
    
    # Check validity
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-5:
                valid = False
    if not valid:
        r1, s = 1.0, 1.0 / np.sqrt(2)
        points = get_points(r1, s)

    return points.tolist()
