import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 14

def _score_from_angles(angles):
    # angles: array of shape (N,)
    # Convert to points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    
    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback if points are collinear or identical (should not happen with distinct angles)
        return -1.0
        
    if hull_area < 1e-10:
        return -1.0
        
    # Calculate Triangle Areas
    # Vectorized calculation for all triples
    # i, j, k
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def _negative_score(angles):
    return -_score_from_angles(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # Safety margin
    
    # Initial guess: Regular 14-gon
    angles_init = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    best_score = _score_from_angles(angles_init)
    best_angles = angles_init.copy()
    
    # Try a few random starts to avoid local minima, plus the regular one
    starts = [angles_init]
    for _ in range(5):
        random_angles = np.sort(np.random.uniform(0, 2*np.pi, N))
        starts.append(random_angles)
        
    for init in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_negative_score, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_angles = res.x
        except Exception:
            pass
            
    # Final check time
    if time.time() > deadline - 5:
         # Return best found so far
         pass
    # Convert best angles to points
    x = np.cos(best_angles)
    y = np.sin(best_angles)
    points = np.column_stack((x, y))
    
    return points
