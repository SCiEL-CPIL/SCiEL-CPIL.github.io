import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0

def min_triangle_area(points):
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Shoelace formula
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                    (points[k][0] - points[i][0]) * (points[j][1] - points[i][1])
                )
                if area < min_a:
                    min_a = area
    return min_a

def is_inside(points, eps=1e-9):
    for p in points:
        x, y = p
        if y < -eps or y > S3 * x + eps or y > S3 * (1 - x) + eps:
            return False
    return True

def objective(flat_points):
    points = flat_points.reshape(N, 2)
    # Penalty for being outside
    pen = 0.0
    for p in points:
        x, y = p
        if y < 0:
            pen += abs(y)
        if y > S3 * x:
            pen += (y - S3 * x)
        if y > S3 * (1 - x):
            pen += (y - S3 * (1 - x))
    
    if pen > 1e-6:
        return 100.0 + pen # Large cost
    
    area = min_triangle_area(points)
    return -area # Minimize negative area

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    bounds = [(0, 1), (0, H)] * N
    
    # Initial population strategy: Random
    # DE parameters
    # popsize=20, maxiter=100 might be too slow?
    # Let's try popsize=15, maxiter=50
    # tolerance=1e-4
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            polish=True, # Use Nelder-Mead for polishing
            mutation=(0.5, 1.5),
            recombination=0.7,
            init='latinhypercube'
        )
        
        best_points = res.x.reshape(N, 2)
        
        # If time allows, do a few more local optimizations from best point
        # and maybe a few random restarts with local optimization
        
        # Local refinement
        if time.time() < deadline - 5.0:
            def local_obj(flat):
                pts = flat.reshape(N, 2)
                if not is_inside(pts, 1e-8):
                    return 100.0
                return -min_triangle_area(pts)
            
            # Nelder-Mead is good for non-smooth
            res_local = minimize(
                local_obj, 
                res.x, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_local.fun < res.fun:
                best_points = res_local.x.reshape(N, 2)
                
        # One more restart if time permits
        if time.time() < deadline - 10.0:
             # Try a random start with local opt
             rng = np.random.default_rng(123)
             # Generate random points inside triangle
             # Use rejection or barycentric
             # Barycentric: u, v, w >= 0, u+v+w=1
             # P = u*A + v*B + w*C
             # u,v,w ~ Dirichlet(1,1,1)
             uvw = rng.dirichlet([1, 1, 1], size=N)
             # A=(0,0), B=(1,0), C=(0.5, H)
             p_rand = uvw[:, 1:2] * np.array([[1, 0]]) + uvw[:, 2:3] * np.array([[0.5, H]])
             # A is 0 so term vanishes
             
             res_local2 = minimize(
                 local_obj, 
                 p_rand.flatten(), 
                 method='Nelder-Mead', 
                 options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
             )
             if res_local2.fun < res_local.fun:
                 best_points = res_local2.x.reshape(N, 2)
                 
        return best_points
    except Exception as e:
        # Fallback: return random valid points
        rng = np.random.default_rng(0)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1:
                # Map to triangle
                # Standard mapping: x = u + v/2, y = v * sqrt(3)/2
                # This maps square [0,1]x[0,1] with u+v<1 to triangle
                x = u + 0.5 * v
                y = H * v
                pts.append([x, y])
        return np.array(pts)
