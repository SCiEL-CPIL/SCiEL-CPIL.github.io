import numpy as np
import scipy.optimize as opt
import time

N = 11
SQRT3 = np.sqrt(3)
TRI_HEIGHT = SQRT3 / 2
TRI_AREA = TRI_HEIGHT / 2.0 # Area of triangle with base 1, height sqrt(3)/2 is sqrt(3)/4. Wait.
# Base is 1 (from (0,0) to (1,0)). Height is sqrt(3)/2.
# Area = 0.5 * base * height = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4.
# The task says "unit-area triangle" in the text but the interface specifies vertices (0,0), (1,0), (1/2, sqrt(3)/2).
# This triangle has area sqrt(3)/4.
# "The notebook uses an equilateral triangle and divides by its area."
# The evaluator measures the raw value.
# So we just need to maximize the minimum area in this specific triangle.

def triangle_area(p1, p2, p3):
    # p1, p2, p3 are (x, y) arrays
    # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

def is_inside_triangle(p):
    x, y = p
    # y >= 0
    if y < 0: return False
    # y <= sqrt(3) * x
    if y > SQRT3 * x + 1e-9: return False
    # y <= sqrt(3) * (1 - x)
    if y > SQRT3 * (1 - x) + 1e-9: return False
    return True

def penalty_for_point(p):
    x, y = p
    d1 = max(0, -y)
    d2 = max(0, y - SQRT3 * x)
    d3 = max(0, y - SQRT3 * (1 - x))
    # Use squared distance for smoothness
    return d1**2 + d2**2 + d3**2

def objective(params, tau):
    points = params.reshape(N, 2)
    
    # Penalty for being outside
    pen = 0.0
    for p in points:
        pen += penalty_for_point(p)
    
    if pen > 1e-6:
        return 1000.0 * pen - 0.1 # Strong penalty
    
    # Calculate all triangle areas
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                a = triangle_area(points[i], points[j], points[k])
                areas.append(a)
    
    areas = np.array(areas)
    
    # Soft min
    # min(areas) approx -1/tau * log(sum(exp(-tau * areas)))
    # To avoid overflow, shift by min
    min_a = np.min(areas)
    exp_terms = np.exp(-tau * (areas - min_a))
    soft_min = min_a - (1.0/tau) * np.log(np.sum(exp_terms))
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    # 1. Random
    # 2. Grid-like
    # 3. Known seed (if available, but we don't have one, so use random + local opt)
    
    # Use multiple restarts
    num_restarts = 20
    
    # Annealing tau
    # Start with high tau (smooth), end with low tau (sharp)
    taus = [10.0, 5.0, 2.0, 1.0, 0.5]
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Generate initial points
        # Try to spread them out
        rng = np.random.RandomState(r)
        points = np.zeros((N, 2))
        # Simple random sampling inside triangle
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = TRI_HEIGHT * v
                    if is_inside_triangle([x, y]):
                        points[i] = [x, y]
                        break
        
        # Optimization loop with annealing
        current_points = points.copy()
        
        for tau in taus:
            if time.time() > deadline:
                break
                
            # Define objective for this tau
            def obj_func(p):
                return objective(p, tau)
            
            # Bounds for L-BFGS-B
            bounds = [(0, 1), (0, TRI_HEIGHT)] * N
            
            try:
                res = opt.minimize(obj_func, current_points.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
                if res.success or res.fun < objective(current_points.flatten(), tau):
                    current_points = res.x.reshape(N, 2)
            except Exception:
                pass
            
            # Check if points are still valid
            valid = all(is_inside_triangle(p) for p in current_points)
            if not valid:
                # Re-project or discard? For now, if invalid, try to fix by projecting
                for i in range(N):
                    p = current_points[i]
                    # Simple projection: clip to box, then check triangle
                    # If still outside, move to centroid
                    if not is_inside_triangle(p):
                         # Move towards centroid
                         centroid = np.array([0.5, TRI_HEIGHT/3])
                         current_points[i] = centroid + 0.5 * (p - centroid)
                         # If still outside, just use centroid
                         if not is_inside_triangle(current_points[i]):
                             current_points[i] = centroid
                             
            # Evaluate true score
            score = -objective(current_points.flatten(), 0.001) # Use very small tau for evaluation (approx min)
            # Actually, just compute the true min area
            areas = []
            for i in range(N):
                for j in range(i+1, N):
                    for k in range(j+1, N):
                        areas.append(triangle_area(current_points[i], current_points[j], current_points[k]))
            true_min = np.min(areas)
            
            if true_min > best_score:
                best_score = true_min
                best_points = current_points.copy()
                
    if best_points is None:
        # Fallback to random
        points = np.random.rand(N, 2)
        for i in range(N):
             while True:
                 u = np.random.random()
                 v = np.random.random()
                 if u + v < 1:
                     points[i] = [u + 0.5 * v, TRI_HEIGHT * v]
                     break
        best_points = points
        best_score = 0.0

    return best_points

# EVOLVE-BLOCK-END
