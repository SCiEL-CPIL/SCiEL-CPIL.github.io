import numpy as np
import time
import random

N = 11

def construct():
    start_time = time.time()
    time_limit = 120.0  # Leave margin for 150s limit
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def inside_triangle(p):
        # Check if point p is inside triangle ABC
        # Using barycentric coordinates
        # Area of ABC
        area_abc = 0.5 * np.abs(np.cross(B-A, C-A))
        
        p1 = np.array([p[0], p[1]])
        area_pbc = 0.5 * np.abs(np.cross(B-p1, C-p1))
        area_pca = 0.5 * np.abs(np.cross(C-p1, A-p1))
        area_pab = 0.5 * np.abs(np.cross(A-p1, B-p1))
        
        # Sum of areas should equal area_abc
        # Also, points must be on the same side of edges
        # Simpler check:
        # Barycentric coords
        v0 = B - A
        v1 = C - A
        v2 = p1 - A
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return False
            
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return u >= -1e-9 and v >= -1e-9 and w >= -1e-9

    def min_area(points):
        # points: (N, 2) array
        min_a = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Area = 0.5 * |cross(p2-p1, p3-p1)|
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    a = 0.5 * abs(cross)
                    if a < min_a:
                        min_a = a
        return min_a

    def objective(points):
        # Returns -min_area for minimization
        # Check validity
        for p in points:
            if not inside_triangle(p):
                return 1e6  # Large penalty
        return -min_area(points)

    # Initialization strategies
    def init_grid(m):
        pts = []
        for i in range(m+1):
            for j in range(m+1-i):
                u = i / m
                v = j / m
                w = 1 - u - v
                # Convert barycentric (u,v,w) to cartesian
                # P = u*A + v*B + w*C
                p = u*A + v*B + w*C
                pts.append(p)
        return np.array(pts)

    def init_random(seed):
        rng = random.Random(seed)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = u*B + v*C # Wait, barycentric mapping
                # Standard mapping: P = (1-u-v)A + uB + vC
                # Since A=0, P = uB + vC
                p = u*B + v*C
                pts.append(p)
        return np.array(pts)

    # Generate initial candidates
    candidates = []
    
    # 1. Grid M=4 (15 points), take first 11
    grid_pts = init_grid(4)
    if len(grid_pts) >= N:
        candidates.append(grid_pts[:N].copy())
        
    # 2. Grid M=3 (10 points) + 1 random center-ish
    grid_pts_3 = init_grid(3)
    if len(grid_pts_3) == N - 1:
        extra = np.array([0.5, 0.2]) # Rough center
        candidates.append(np.vstack([grid_pts_3, extra]))
        
    # 3. Random
    candidates.append(init_random(1))
    candidates.append(init_random(2))

    best_score = -1e9
    best_points = None

    for idx, init_pts in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        current_points = init_pts.copy()
        
        # Simulated Annealing
        T = 0.05
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.05
        
        current_obj = objective(current_points)
        best_local_obj = current_obj
        best_local_points = current_points.copy()
        
        max_iterations = 20000
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            if T < T_min:
                break
                
            # Pick random point
            i = random.randint(0, N-1)
            old_p = current_points[i].copy()
            
            # Perturb
            new_p = old_p + step_size * np.array([random.gauss(0,1), random.gauss(0,1)])
            
            # Check bounds
            if not inside_triangle(new_p):
                # Reject or push back? Reject is simpler for SA
                continue
                
            current_points[i] = new_p
            new_obj = objective(current_points)
            
            delta = new_obj - current_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_obj = new_obj
                if new_obj < best_local_obj:
                    best_local_obj = new_obj
                    best_local_points = current_points.copy()
            else:
                current_points[i] = old_p
                
            T *= cooling
            
        # Polish with Nelder-Mead
        if time.time() - start_time < time_limit - 10:
            try:
                from scipy.optimize import minimize
                res = minimize(objective, best_local_points, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.success or res.fun < best_local_obj:
                    best_local_obj = res.fun
                    best_local_points = res.x
            except Exception:
                pass

        # Update global best
        # objective returns -min_area, so lower is better
        if best_local_obj < best_score: # Wait, best_score is initialized to -1e9? 
             # If objective is -min_area, we want to MINIMIZE objective.
             # So best_score should be the minimum objective value found.
             # Let's rename to best_obj_val
             pass

    # Re-evaluate logic
    best_obj_val = 1e9 # We want to minimize objective (which is -min_area)
    best_pts = None
    
    for idx, init_pts in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        current_points = init_pts.copy()
        
        # Simulated Annealing
        T = 0.05
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.05
        
        current_obj = objective(current_points)
        best_local_obj = current_obj
        best_local_points = current_points.copy()
        
        max_iterations = 20000
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
                
            if T < T_min:
                break
                
            i = random.randint(0, N-1)
            old_p = current_points[i].copy()
            new_p = old_p + step_size * np.array([random.gauss(0,1), random.gauss(0,1)])
            
            if not inside_triangle(new_p):
                continue
                
            current_points[i] = new_p
            new_obj = objective(current_points)
            
            delta = new_obj - current_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_obj = new_obj
                if new_obj < best_local_obj:
                    best_local_obj = new_obj
                    best_local_points = current_points.copy()
            else:
                current_points[i] = old_p
                
            T *= cooling
            
        # Polish
        if time.time() - start_time < time_limit - 10:
            try:
                from scipy.optimize import minimize
                res = minimize(objective, best_local_points, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_local_obj:
                    best_local_obj = res.fun
                    best_local_points = res.x
            except Exception:
                pass

        if best_local_obj < best_obj_val:
            best_obj_val = best_local_obj
            best_pts = best_local_points.copy()

    if best_pts is None:
        # Fallback to seed program logic
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
        
    return best_pts

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    time_limit = 120.0
    
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def inside_triangle(p):
        v0 = B - A
        v1 = C - A
        v2 = p - A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return False
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return u >= -1e-9 and v >= -1e-9 and w >= -1e-9

    def min_area(points):
        min_a = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = v1[0]*v2[1] - v1[1]*v2[0]
                    a = 0.5 * abs(cross)
                    if a < min_a:
                        min_a = a
        return min_a

    def objective(points):
        for p in points:
            if not inside_triangle(p):
                return 1e6
        return -min_area(points)

    def init_grid(m):
        pts = []
        for i in range(m+1):
            for j in range(m+1-i):
                u = i / m
                v = j / m
                p = u*B + v*C
                pts.append(p)
        return np.array(pts)

    def init_random(seed):
        rng = random.Random(seed)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                p = u*B + v*C
                pts.append(p)
        return np.array(pts)

    candidates = []
    grid_pts = init_grid(4)
    if len(grid_pts) >= N:
        candidates.append(grid_pts[:N].copy())
        
    grid_pts_3 = init_grid(3)
    if len(grid_pts_3) == N - 1:
        extra = np.array([0.5, 0.2])
        candidates.append(np.vstack([grid_pts_3, extra]))
        
    candidates.append(init_random(1))
    candidates.append(init_random(2))

    best_obj_val = 1e9
    best_pts = None

    for idx, init_pts in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
            
        current_points = init_pts.copy()
        
        T = 0.05
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.05
        
        current_obj = objective(current_points)
        best_local_obj = current_obj
        best_local_points = current_points.copy()
        
        max_iterations = 20000
        
        for it in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            if T < T_min:
                break
                
            i = random.randint(0, N-1)
            old_p = current_points[i].copy()
            new_p = old_p + step_size * np.array([random.gauss(0,1), random.gauss(0,1)])
            
            if not inside_triangle(new_p):
                continue
                
            current_points[i] = new_p
            new_obj = objective(current_points)
            
            delta = new_obj - current_obj
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_obj = new_obj
                if new_obj < best_local_obj:
                    best_local_obj = new_obj
                    best_local_points = current_points.copy()
            else:
                current_points[i] = old_p
                
            T *= cooling
            
        if time.time() - start_time < time_limit - 10:
            try:
                res = minimize(objective, best_local_points, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_local_obj:
                    best_local_obj = res.fun
                    best_local_points = res.x
            except Exception:
                pass

        if best_local_obj < best_obj_val:
            best_obj_val = best_local_obj
            best_pts = best_local_points.copy()

    if best_pts is None:
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
        
    return best_pts
# EVOLVE-BLOCK-END
