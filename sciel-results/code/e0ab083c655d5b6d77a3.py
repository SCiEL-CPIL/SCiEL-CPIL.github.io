import numpy as np
import time

N = 11
H = np.sqrt(3) / 2

def random_point_in_triangle():
    r1 = np.random.rand()
    r2 = np.random.rand()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    x = r1 * 1.0 + r2 * 0.5
    y = r2 * H
    return np.array([x, y])

def min_triangle_area(points):
    min_a = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_a:
                    min_a = area
    return min_a

def is_inside_triangle(p):
    v0 = np.array([0.5, H])
    v1 = np.array([1.0, 0.0])
    v2 = p
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d02 = np.dot(v0, v2)
    d12 = np.dot(v1, v2)
    denom = d00 * d11 - d01 * d01
    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom
    
    return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Generate 15-point lattice for initialization
    k = 5
    points_15 = []
    for j in range(k):
        for i in range(k - j):
            x = (i + 0.5 * j) / (k - 1) * 1.0
            y = j * H / (k - 1) * 1.0
            points_15.append([x, y])
    points_15 = np.array(points_15)
    
    best_points_global = None
    best_score_global = -float('inf')
    
    # Run multiple SA instances if time permits
    # Let's run one long SA with restarts, or multiple short ones.
    # Multiple short ones allows better parallelization of "ideas" (different starts).
    
    num_runs = 5
    
    for run in range(num_runs):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        if run == 0:
            # Start with best lattice subset (heuristic: random subset)
            init_indices = np.random.choice(15, 11, replace=False)
            current_points = points_15[init_indices].copy()
        else:
            # Random start
            current_points = np.array([random_point_in_triangle() for _ in range(N)])
            # Ensure they are distinct enough? Not strictly necessary, SA will move them.
            
        current_score = min_triangle_area(current_points)
        best_points_local = current_points.copy()
        best_score_local = current_score
        
        T = 0.05
        T_min = 1e-7
        alpha = 0.998
        max_iter = 50000
        
        stuck_counter = 0
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            idx = np.random.randint(N)
            original_point = current_points[idx].copy()
            
            # Move generation
            if np.random.rand() < 0.05:
                # Large jump: random point in triangle
                new_point = random_point_in_triangle()
            else:
                # Small step
                # Step size proportional to T?
                step_size = 0.05 * (T / 0.05) + 0.005
                delta = np.random.normal(0, step_size, 2)
                new_point = original_point + delta
                if not is_inside_triangle(new_point):
                    # Try to project or just skip
                    # Let's try a few times to find a valid neighbor
                    found = False
                    for _ in range(5):
                        delta = np.random.normal(0, step_size, 2)
                        new_point = original_point + delta
                        if is_inside_triangle(new_point):
                            found = True
                            break
                    if not found:
                        continue
            
            current_points[idx] = new_point
            new_score = min_triangle_area(current_points)
            
            delta_score = new_score - current_score
            
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
                current_score = new_score
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_points_local = current_points.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                current_points[idx] = original_point
                stuck_counter += 1
            
            # Restart logic: if stuck for a long time, reset to best local with perturbation
            if stuck_counter > 5000:
                current_points = best_points_local.copy()
                # Perturb best points slightly
                for p_idx in range(N):
                    if np.random.rand() < 0.5:
                        delta = np.random.normal(0, 0.05, 2)
                        new_p = current_points[p_idx] + delta
                        if is_inside_triangle(new_p):
                            current_points[p_idx] = new_p
                current_score = min_triangle_area(current_points)
                T = max(T, 0.02) # Reheat
                stuck_counter = 0
            
            T *= alpha
            if T < T_min:
                T = T_min
        
        if best_score_local > best_score_global:
            best_score_global = best_score_local
            best_points_global = best_points_local.copy()
            
    return best_points_global
