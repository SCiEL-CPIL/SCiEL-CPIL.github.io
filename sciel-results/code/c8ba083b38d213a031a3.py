import numpy as np
import time
from scipy.spatial.distance import pdist

N = 300

def construct():
    # 1. Initialization: Hexagonal lattice
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3))) * 1.05
    
    points_list = []
    k = int(1.0 / s_init) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s_init
            y = j * s_init * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    if len(points) > N:
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        points = points[indices]
    elif len(points) < N:
        # Adjust spacing to get enough points
        while len(points) < N:
            s_init *= 0.95
            points_list = []
            k = int(1.0 / s_init) + 2
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * s_init
                    y = j * s_init * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        points_list.append([x, y])
            points = np.array(points_list)
            if len(points) > N:
                dists = np.linalg.norm(points, axis=1)
                indices = np.argsort(dists)[:N]
                points = points[indices]
            break
            
    points = points.astype(np.float64)
    
    def calc_radius(pts):
        dists_center = np.linalg.norm(pts, axis=1)
        r_bound = 1.0 - np.max(dists_center)
        
        dists = pdist(pts)
        min_pair_dist = np.min(dists)
        r_pair = min_pair_dist / 2.0
        
        return min(r_bound, r_pair)

    current_r = calc_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    T = 0.005
    T_min = 0.000001
    alpha = 0.9995
    step_size = 0.05
    
    start_time = time.time()
    time_limit = 115.0
    
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(N)
        
        delta = np.random.normal(0, step_size, 2)
        
        new_pts = points.copy()
        new_pts[i] += delta
        
        # Quick check: if point goes far outside, skip
        if np.linalg.norm(new_pts[i]) > 1.2:
            continue
            
        new_r = calc_radius(new_pts)
        
        if new_r > current_r:
            points = new_pts
            current_r = new_r
            no_improve = 0
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            prob = np.exp((new_r - current_r) / T)
            if np.random.rand() < prob:
                points = new_pts
                current_r = new_r
            no_improve += 1
            
        T *= alpha
        if T < T_min:
            T = T_min
            # If T is very low and we are stuck, maybe restart with higher T?
            if no_improve > 5000:
                T = 0.005
                step_size = 0.08
                no_improve = 0
                # Optional: reset to best
                # points = best_points.copy()
                # current_r = best_r

    return best_points
