import numpy as np
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 30
    
    # Pre-compute triple indices for 30 points
    def _get_triple_indices(n):
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        return np.array(i_idx, dtype=np.int32), np.array(j_idx, dtype=np.int32), np.array(k_idx, dtype=np.int32)
    
    _TRIPLE_I, _TRIPLE_J, _TRIPLE_K = _get_triple_indices(n)
    _N_TRIPLES = len(_TRIPLE_I)
    
    def _vectorized_compute_areas(points):
        """
        Compute absolute doubled areas for all triples of 30 points.
        points: numpy array of shape (30, 2)
        Returns: numpy array of shape (4500,) with absolute determinants.
        """
        pi = points[_TRIPLE_I]
        pj = points[_TRIPLE_J]
        pk = points[_TRIPLE_K]
        dx_ji = pj[:, 0] - pi[:, 0]
        dy_ji = pj[:, 1] - pi[:, 1]
        dx_ki = pk[:, 0] - pi[:, 0]
        dy_ki = pk[:, 1] - pi[:, 1]
        dets = dx_ji * dy_ki - dy_ji * dx_ki
        return np.abs(dets)
    
    def _get_max_multiplicity(areas):
        """
        Given an array of areas, return (max_count, most_frequent_area).
        Ignores zero areas.
        """
        if len(areas) == 0:
            return 0, 0
        non_zero = areas[areas > 0]
        if len(non_zero) == 0:
            return 0, 0
        unique_areas, counts = np.unique(non_zero, return_counts=True)
        max_idx = np.argmax(counts)
        return int(counts[max_idx]), int(unique_areas[max_idx])
    
    def _count_area(areas, target):
        """Count how many times target area appears in areas."""
        return int(np.sum(areas == target))
    
    def _incremental_move(areas, points, idx):
        """
        Update the areas array after moving point idx.
        Recomputes only triples involving idx.
        """
        mask_i = _TRIPLE_I == idx
        mask_j = _TRIPLE_J == idx
        mask_k = _TRIPLE_K == idx
        mask = mask_i | mask_j | mask_k
        
        if np.any(mask):
            ti = _TRIPLE_I[mask]
            tj = _TRIPLE_J[mask]
            tk = _TRIPLE_K[mask]
            
            pi = points[ti]
            pj = points[tj]
            pk = points[tk]
            
            dx_ji = pj[:, 0] - pi[:, 0]
            dy_ji = pj[:, 1] - pi[:, 1]
            dx_ki = pk[:, 0] - pi[:, 0]
            dy_ki = pk[:, 1] - pi[:, 1]
            
            new_dets = dx_ji * dy_ki - dy_ji * dx_ki
            areas[mask] = np.abs(new_dets)
        
        return areas
    
    def _sa_search(points, deadline):
        """
        Simulated annealing with incremental updates.
        Returns (best_score, best_points).
        """
        curr_points = points.copy()
        curr_occupied = set(map(tuple, curr_points.tolist()))
        
        curr_areas = _vectorized_compute_areas(curr_points)
        
        curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
        target_area = curr_max_area
        curr_target_count = _count_area(curr_areas, target_area)
        
        best_score = max(curr_target_count, curr_max_mult)
        best_points = curr_points.copy()
        
        temp = 3.0
        cooling = 0.99995
        min_temp = 0.003
        
        no_improve_count = 0
        max_no_improve = 5000
        steps = 0
        last_recompute = 0
        recompute_interval = 1500
        
        while time.time() < deadline:
            steps += 1
            
            # Pick a random point to move
            idx = random.randint(0, 29)
            old_x, old_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
            
            # Generate new position
            r = random.random()
            if r < 0.35:
                new_x = random.randint(-30, 30)
                new_y = random.randint(-30, 30)
            elif r < 0.75:
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                new_x = max(-30, min(30, new_x))
                new_y = max(-30, min(30, new_y))
            else:
                new_x = old_x + random.randint(-8, 8)
                new_y = old_y + random.randint(-8, 8)
                new_x = max(-30, min(30, new_x))
                new_y = max(-30, min(30, new_y))
            
            new_point = (new_x, new_y)
            if new_point == (old_x, old_y):
                continue
            if new_point in curr_occupied:
                continue
            
            # Perform move
            curr_points[idx] = [new_x, new_y]
            curr_occupied.remove((old_x, old_y))
            curr_occupied.add(new_point)
            
            # Incremental update
            _incremental_move(curr_areas, curr_points, idx)
            
            # Recompute full areas periodically to avoid drift
            if steps - last_recompute >= recompute_interval:
                curr_areas = _vectorized_compute_areas(curr_points)
                last_recompute = steps
            
            # Evaluate
            new_max_mult, new_max_area = _get_max_multiplicity(curr_areas)
            
            # Dynamic target switching
            if new_max_area != target_area:
                target_area = new_max_area
                curr_target_count = _count_area(curr_areas, target_area)
            
            new_target_count = _count_area(curr_areas, target_area)
            
            new_score = max(new_target_count, new_max_mult)
            curr_score = max(curr_target_count, curr_max_mult)
            
            improved = False
            if new_score > curr_score:
                improved = True
                no_improve_count = 0
            elif new_score == curr_score:
                if new_target_count > curr_target_count:
                    improved = True
                    no_improve_count = 0
                elif random.random() < 0.5:
                    improved = True
            else:
                delta = curr_score - new_score
                if random.random() < np.exp(-delta / max(temp, 1e-10)):
                    improved = True
            
            if improved:
                curr_target_count = new_target_count
                curr_max_mult = new_max_mult
                curr_max_area = new_max_area
                curr_score = new_score
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = curr_points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    
                temp = max(min_temp, temp * cooling)
            else:
                # Revert
                curr_points[idx] = [old_x, old_y]
                curr_occupied.remove(new_point)
                curr_occupied.add((old_x, old_y))
                _incremental_move(curr_areas, curr_points, idx)
                
                no_improve_count += 1
                temp = max(min_temp, temp * cooling)
            
            if no_improve_count > max_no_improve:
                break
        
        return best_score, best_points
    
    def _hill_climb(points, deadline):
        """
        Local hill climbing with vectorized area computation.
        Returns (best_score, best_points).
        """
        curr_points = points.copy()
        curr_occupied = set(map(tuple, curr_points.tolist()))
        
        curr_areas = _vectorized_compute_areas(curr_points)
        curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
        curr_score = curr_max_mult
        
        best_score = curr_score
        best_points = curr_points.copy()
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(n):
                orig_x, orig_y = int(curr_points[idx, 0]), int(curr_points[idx, 1])
                
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                            continue
                        new_pos = (new_x, new_y)
                        if new_pos in curr_occupied:
                            continue
                        
                        # Perform move
                        curr_points[idx] = [new_x, new_y]
                        curr_occupied.remove((orig_x, orig_y))
                        curr_occupied.add(new_pos)
                        
                        # Incremental update
                        _incremental_move(curr_areas, curr_points, idx)
                        
                        # Evaluate
                        new_max_mult, _ = _get_max_multiplicity(curr_areas)
                        
                        if new_max_mult > curr_score:
                            curr_score = new_max_mult
                            improved = True
                            if curr_score > best_score:
                                best_score = curr_score
                                best_points = curr_points.copy()
                        else:
                            # Revert
                            curr_points[idx] = [orig_x, orig_y]
                            curr_occupied.remove(new_pos)
                            curr_occupied.add((orig_x, orig_y))
                            _incremental_move(curr_areas, curr_points, idx)
                        
                        if time.time() > deadline:
                            return best_score, best_points
                            
                if time.time() > deadline:
                    return best_score, best_points
        
        return best_score, best_points
    
    # Generate multiple initial configurations
    def _grid_points(rows, cols, offset_x=0, offset_y=0, spacing=1):
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = i * spacing + offset_x
                y = j * spacing + offset_y
                pts.append((x, y))
        return np.array(pts[:30], dtype=np.int32)
    
    def _sheared_grid(rows, cols, shear, offset_x=0, offset_y=0, spacing=1):
        pts = []
        for i in range(rows):
            for j in range(cols):
                x = i * spacing + int(shear * j) + offset_x
                y = j * spacing + offset_y
                pts.append((x, y))
        return np.array(pts[:30], dtype=np.int32)
    
    def _random_points_in_box(n=30, box=30):
        points = set()
        while len(points) < n:
            x = random.randint(-box, box)
            y = random.randint(-box, box)
            points.add((x, y))
        return np.array(list(points), dtype=np.int32)
    
    # Initial configurations
    configs = []
    configs.append(_grid_points(6, 5, -5, -4))
    configs.append(_grid_points(5, 6, -4, -5))
    configs.append(_sheared_grid(6, 5, 1, -5, -4))
    configs.append(_sheared_grid(6, 5, -1, -5, -4))
    configs.append(_sheared_grid(5, 6, 1, -4, -5))
    configs.append(_sheared_grid(5, 6, -1, -4, -5))
    configs.append(_random_points_in_box())
    configs.append(_random_points_in_box())
    
    # Evaluate all initial configs and pick the best
    best_initial_score = 0
    best_initial_points = configs[0]
    
    for cfg in configs:
        areas = _vectorized_compute_areas(cfg)
        score, _ = _get_max_multiplicity(areas)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_points = cfg
    
    overall_best_score = best_initial_score
    overall_best_points = best_initial_points
    
    # Run hill climbing on the best initial config
    hc_score, hc_points = _hill_climb(best_initial_points, deadline)
    if hc_score > overall_best_score:
        overall_best_score = hc_score
        overall_best_points = hc_points
    
    # Run simulated annealing on the best config found so far
    if time.time() < deadline:
        sa_score, sa_points = _sa_search(overall_best_points, deadline)
        if sa_score > overall_best_score:
            overall_best_score = sa_score
            overall_best_points = sa_points
    
    # Final polish with hill climbing
    if time.time() < deadline - 5:
        final_score, final_points = _hill_climb(overall_best_points, deadline)
        if final_score > overall_best_score:
            overall_best_score = final_score
            overall_best_points = final_points
    
    # Return as list of tuples
    return [tuple(p) for p in overall_best_points]
