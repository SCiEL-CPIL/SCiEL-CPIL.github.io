import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
# A = (0,0), B = (1,0), C = (0.5, sqrt(3)/2)
# Area = sqrt(3)/4

def get_min_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (N, 2) array
    """
    if points.shape != (N, 2):
        return 0.0
    
    # Check if all points are inside the triangle
    # Triangle inequalities for point (x, y):
    # 1. y >= 0
    # 2. x >= 0
    # 3. x <= 1
    # 4. y <= sqrt(3) * x
    # 5. y <= -sqrt(3) * (x - 1)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Simple bounds check
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > np.sqrt(3)/2):
        return 0.0
        
    # Slope checks
    # Line AC: y = sqrt(3) x => y - sqrt(3)x <= 0
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return 0.0
    # Line BC: y = -sqrt(3)(x-1) => y + sqrt(3)x - sqrt(3) <= 0
    if np.any(y > -np.sqrt(3) * (x - 1) + 1e-9):
        return 0.0

    # Calculate areas for all combinations of 3 points
    # N=11, so 165 triangles
    indices = np.arange(N)
    min_area = float('inf')
    
    # Vectorized calculation might be faster, but 165 is small.
    # Let's use combinations for clarity and safety, or vectorize.
    # Vectorized:
    # Get all pairs of indices
    # Actually, let's just loop, it's fast enough for 165.
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-6: # Early exit if points are too close
                        return 0.0
                        
    return min_area

def objective(x):
    """
    Objective function for minimization.
    Minimizes -min_area.
    """
    points = x.reshape(N, 2)
    ma = get_min_area(points)
    if ma == 0.0:
        # Penalize invalid points heavily
        return 1000.0 
    return -ma

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    lb = np.zeros(N * 2)
    ub = np.zeros(N * 2)
    for i in range(N):
        lb[i*2] = 0.0
        lb[i*2+1] = 0.0
        ub[i*2] = 1.0
        ub[i*2+1] = np.sqrt(3)/2
        
    # Initial random guess to ensure we have a valid starting point if DE fails or for NM
    # Generate a random valid point
    def random_valid_point():
        while True:
            u = np.random.random(2)
            x, y = u
            if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1) and x<=1 and x>=0 and y>=0:
                return np.array([x,y])
                
    # Simple random initialization for DE bounds is handled by DE, 
    # but we need a good initial guess for NM.
    # Let's use a grid or random points for NM start.
    
    best_x = None
    best_val = -1000.0
    
    # Run Differential Evolution
    # popsize=15 is default. maxiter=50.
    # tol=0.01
    # We can use a callback to stop if time is up.
    def callback(xk, convergence):
        if time.time() > deadline - 20: # Leave 20s for NM
            return True
        return False

    try:
        res = differential_evolution(objective, bounds=list(zip(lb, ub)), 
                                      maxiter=100, popsize=20, 
                                      tol=1e-4, callback=callback,
                                      seed=42)
        
        if res.success or res.fun < 1000.0: # res.fun < 1000 means it found a valid point
            best_x = res.x
            best_val = -res.fun
    except Exception as e:
        pass
    
    # If DE didn't find a good solution, try a random valid configuration
    if best_x is None or best_val < 0.01:
         # Generate 11 random points inside triangle
         pts = np.array([random_valid_point() for _ in range(N)])
         best_x = pts.flatten()
         best_val = get_min_area(pts)
         
    # Refine with Nelder-Mead
    if time.time() < deadline - 5:
        # NM minimizes. We want to maximize min_area, so minimize -min_area.
        # But objective(x) returns -ma for valid, 1000 for invalid.
        # This is non-smooth at the boundary. NM might struggle with the hard penalty.
        # However, if we start from a good DE point, it should stay inside.
        
        try:
            res_nm = minimize(objective, best_x, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_nm.fun < 1000.0:
                val_nm = -res_nm.fun
                if val_nm > best_val:
                    best_x = res_nm.x
                    best_val = val_nm
        except Exception:
            pass
            
    # Final check and return
    points = best_x.reshape(N, 2)
    # Ensure validity one last time
    if get_min_area(points) > 0:
        return points
    else:
        # Fallback to a known valid configuration if something went wrong
        # E.g. vertices and center
        pts = np.array([
            [0,0], [1,0], [0.5, np.sqrt(3)/2],
            [0.5, np.sqrt(3)/6], # Centroid
            [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4],
            [0.5, 0], [0.25, 0], [0.75, 0],
            [0.5, np.sqrt(3)/2 - 0.1], [0.1, 0.1]
        ])
        # This fallback might not be valid or optimal, but better than crash
        # Let's just return the random valid points generated earlier if this fails
        # Actually, the random_valid_point logic is inside the if block.
        # Let's just return best_x if it's not None, else random.
        if best_x is not None:
             return best_x.reshape(N, 2)
        return np.array([random_valid_point() for _ in range(N)])
