import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 30
    min_coord = -30
    max_coord = 30
    
    triple_indices = np.array(list(combinations(range(n), 3)))
    i_idx = triple_indices[:, 0]
    j_idx = triple_indices[:, 1]
    k_idx = triple_indices[:, 2]
    
    def calculate_scores(points_np):
        Pi = points_np[i_idx]
        Pj = points_np[j_idx]
        Pk = points_np[k_idx]
        
        areas = np.abs(Pi[:, 0] * (Pj[:, 1] - Pk[:, 1]) + 
                       Pj[:, 0] * (Pk[:, 1] - Pi[:, 1]) + 
                       Pk[:, 0] * (Pi[:, 1] - Pj[:, 1]))
        
        unique_areas, counts = np.unique(areas, return_counts=True)
        mask = unique_areas > 0
        if not np.any(mask):
            return 0, 0
            
        valid_counts = counts[mask]
        max_mult = np.max(valid_counts)
        sum_sq = np.sum(valid_counts ** 2)
        
        return int(max_mult), int(sum_sq)

    def initial_config_np():
        xs = np.array([-15, -10, -5, 0, 5, 10])
        ys = np.array([-10, -5, 0, 5, 10])
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        return points

    best_points = None
    best_max_mult = 0
    best_sum_sq = 0
    
    current_points = initial_config_np()
    current_point_set = set(map(tuple, current_points))
    
    current_max_mult, current_sum_sq = calculate_scores(current_points)
    
    if current_max_mult > best_max_mult:
        best_max_mult = current_max_mult
        best_points = current_points.copy()
        best_sum_sq = current_sum_sq

    T = 10.0
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        idx = random.randint(0, n - 1)
        old_x, old_y = current_points[idx]
        old_point = (int(old_x), int(old_y))
        
        if random.random() < 0.5:
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        else:
            new_x = random.randint(min_coord, max_coord)
            new_y = random.randint(min_coord, max_coord)
            
        new_x = max(min_coord, min(max_coord, new_x))
        new_y = max(min_coord, min(max_coord, new_y))
        new_point = (int(new_x), int(new_y))
        
        if new_point in current_point_set:
            continue
            
        # Apply move
        current_points[idx] = [new_x, new_y]
        current_point_set.remove(old_point)
        current_point_set.add(new_point)
        
        # Evaluate
        new_max_mult, new_sum_sq = calculate_scores(current_points)
        
        # Update best
        if new_max_mult > best_max_mult:
            best_max_mult = new_max_mult
            best_points = current_points.copy()
            best_sum_sq = new_sum_sq
        
        # Acceptance criterion
        delta = new_sum_sq - current_sum_sq
        
        if delta > 0:
            current_sum_sq = new_sum_sq
        else:
            if T > T_min:
                p = math.exp(delta / T)
                if random.random() < p:
                    current_sum_sq = new_sum_sq
                else:
                    # Revert
                    current_points[idx] = [old_x, old_y]
                    current_point_set.remove(new_point)
                    current_point_set.add(old_point)
            else:
                # Revert
                current_points[idx] = [old_x, old_y]
                current_point_set.remove(new_point)
                current_point_set.add(old_point)
                
        T *= cooling_rate
        if T < T_min:
            T = 5.0

    if best_points is None:
        best_points = initial_config_np()
        
    return [(int(x), int(y)) for x, y in best_points]
