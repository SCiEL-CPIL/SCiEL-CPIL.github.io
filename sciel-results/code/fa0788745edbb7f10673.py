import numpy as np
import scipy.optimize as opt
import time

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(s, t):
    # s, t are in [0,1]
    # u1 = s^2
    # u2 = (1-s^2)t^2
    # u0 = (1-s^2)(1-t^2)
    u1 = s**2
    u2 = (1 - s**2) * t**2
    u0 = 1 - u1 - u2
    return u0 * V0 + u1 * V1 + u2 * V2

def cart_to_bary_params(x, y):
    # Inverse of bary_to_cart is complex, not needed if we optimize in s,t space
    pass

def compute_areas(points):
    # points: (N, 2)
    # Returns array of areas for all i<j<k
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                areas.append(area)
    return np.array(areas)

def objective(params, lam):
    # params: (2N,) array of s, t for each point
    points = np.zeros((N, 2))
    for i in range(N):
        s = params[2*i]
        t = params[2*i+1]
        points[i] = bary_to_cart(s, t)
    
    areas = compute_areas(points)
    # Soft min: -1/lam * log(sum(exp(-lam * areas)))
    # To avoid overflow, use log-sum-exp trick
    max_area = np.max(areas)
    log_sum_exp = max_area + np.log(np.sum(np.exp(-lam * (areas - max_area))))
    soft_min = -log_sum_exp / lam
    return -soft_min # We want to maximize soft_min, so minimize negative

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_val = 0.0
    best_params = None
    
    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(10):
        params = np.random.rand(2*N)
        inits.append(params)
        
    # 2. Symmetric / Structured
    # Try to place points near vertices, midpoints, center
    # This is heuristic
    
    # Optimization loop
    lambdas = [10, 50, 100, 500, 1000]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_params = init.copy()
        
        # Annealing
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            # Use Nelder-Mead or Powell
            # Powell is robust for non-smooth
            try:
                res = opt.minimize(objective, current_params, args=(lam,), method='Powell', 
                                   bounds=[(0,1)] * (2*N), 
                                   options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < objective(current_params, lam): # Check improvement
                    current_params = res.x
            except Exception:
                pass
            
        # Evaluate hard min
        points = np.zeros((N, 2))
        for i in range(N):
            s = current_params[2*i]
            t = current_params[2*i+1]
            points[i] = bary_to_cart(s, t)
            
        areas = compute_areas(points)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_params = current_params.copy()
            
    # If we have time, do a final polish with high lambda
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res = opt.minimize(objective, best_params, args=(1000,), method='Nelder-Mead', 
                               bounds=[(0,1)] * (2*N), 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < objective(best_params, 1000):
                best_params = res.x
                points = np.zeros((N, 2))
                for i in range(N):
                    s = best_params[2*i]
                    t = best_params[2*i+1]
                    points[i] = bary_to_cart(s, t)
                areas = compute_areas(points)
                val = np.min(areas)
                if val > best_val:
                    best_val = val
        except Exception:
            pass
            
    # Convert best_params to points
    if best_params is None:
        # Fallback to random
        points = np.random.rand(N, 2)
        # Map to triangle? Just return random cartesian in square for now? 
        # No, must be inside triangle.
        # Use the bary mapping with random s,t
        best_params = np.random.rand(2*N)
        
    points = np.zeros((N, 2))
    for i in range(N):
        s = best_params[2*i]
        t = best_params[2*i+1]
        points[i] = bary_to_cart(s, t)
        
    return points

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(s, t):
    u1 = s**2
    u2 = (1 - s**2) * t**2
    u0 = 1 - u1 - u2
    return u0 * V0 + u1 * V1 + u2 * V2

def compute_areas(points):
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                areas.append(area)
    return np.array(areas)

def objective(params, lam):
    points = np.zeros((N, 2))
    for i in range(N):
        s = params[2*i]
        t = params[2*i+1]
        points[i] = bary_to_cart(s, t)
    
    areas = compute_areas(points)
    if len(areas) == 0:
        return 0.0
    max_area = np.max(areas)
    log_sum_exp = max_area + np.log(np.sum(np.exp(-lam * (areas - max_area))))
    soft_min = -log_sum_exp / lam
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_val = 0.0
    best_params = None
    
    inits = []
    for _ in range(15):
        params = np.random.rand(2*N)
        inits.append(params)
        
    # Add a structured init: vertices and midpoints
    # Vertices: (0,0)->s=0,t=0; (1,0)->s=1,t=0; (0.5, sqrt3/2)->s=0,t=1
    # Midpoints: 
    # This is complex to map directly to s,t. Skip for now, random is usually sufficient with enough restarts.
    
    lambdas = [5, 20, 100, 500]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_params = init.copy()
        
        for lam in lambdas:
            if time.time() > deadline:
                break
                
            try:
                # Use Powell for robustness
                res = opt.minimize(objective, current_params, args=(lam,), method='Powell', 
                                   bounds=[(0,1)] * (2*N), 
                                   options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-3})
                if res.fun < objective(current_params, lam):
                    current_params = res.x
            except Exception:
                pass
            
        points = np.zeros((N, 2))
        for i in range(N):
            s = current_params[2*i]
            t = current_params[2*i+1]
            points[i] = bary_to_cart(s, t)
            
        areas = compute_areas(points)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_params = current_params.copy()
            
    if time.time() < deadline - 5.0 and best_params is not None:
        try:
            res = opt.minimize(objective, best_params, args=(1000,), method='Nelder-Mead', 
                               bounds=[(0,1)] * (2*N), 
                               options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < objective(best_params, 1000):
                best_params = res.x
                points = np.zeros((N, 2))
                for i in range(N):
                    s = best_params[2*i]
                    t = best_params[2*i+1]
                    points[i] = bary_to_cart(s, t)
                areas = compute_areas(points)
                val = np.min(areas)
                if val > best_val:
                    best_val = val
        except Exception:
            pass
            
    if best_params is None:
        best_params = np.random.rand(2*N)
        
    points = np.zeros((N, 2))
    for i in range(N):
        s = best_params[2*i]
        t = best_params[2*i+1]
        points[i] = bary_to_cart(s, t)
        
    return points
# EVOLVE-BLOCK-END
