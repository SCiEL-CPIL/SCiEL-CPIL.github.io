import itertools
import random
import time
import math
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    def initial_grid_5x6():
        pts = []
        for x in range(-2, 3):
            for y in range(-3, 3):
                pts.append((x, y))
        return pts

    def initial_grid_6x5():
        pts = []
        for x in range(-3, 3):
            for y in range(-2, 3):
                pts.append((x, y))
        return pts

    def initial_grid_5x6_shifted():
        pts = []
        for x in range(-2, 3):
            for y in range(-2, 3):
                pts.append((x, y))
        # Add 5 more points
        pts.append((-2, 3))
        pts.append((-1, 3))
        pts.append((0, 3))
        pts.append((1, 3))
        pts.append((2, 3))
        return pts

    def initial_grid_6x5_shifted():
        pts = []
        for x in range(-3, 3):
            for y in range(-2, 3):
                pts.append((x, y))
        # Add 5 more points
        pts.append((-3, 3))
        pts.append((-2, 3))
        pts.append((-1, 3))
        pts.append((0, 3))
        pts.append((1, 3))
        return pts

    def initial_lattice():
        # Try a different lattice structure
        pts = []
        for i in range(5):
            for j in range(6):
                pts.append((i, j))
        return pts

    def initial_hexagonal():
        # Approximate hexagonal packing
        pts = []
        rows = [5, 6, 5, 6, 5, 3]
        for r, count in enumerate(rows):
            offset = r * 0.5
            for c in range(count):
                x = int(round(c + offset))
                y = r
                pts.append((x, y))
        # Ensure distinct and in bounds
        seen = set()
        unique_pts = []
        for p in pts:
            if p not in seen and -30 <= p[0] <= 30 and -30 <= p[1] <= 30:
                seen.add(p)
                unique_pts.append(p)
        while len(unique_pts) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in seen:
                seen.add((x, y))
                unique_pts.append((x, y))
        return unique_pts[:30]

    def calculate_area_counts(points):
        """Return dict of area -> count for all positive areas"""
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        counts = {}
        p = pts
        
        for i in range(n - 2):
            dx = p[i+1:, 0] - p[i, 0]
            dy = p[i+1:, 1] - p[i, 1]
            
            det_matrix = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            
            rows, cols = np.triu_indices(n - i - 1, k=1)
            dets = det_matrix[rows, cols]
            
            nonzero_dets = dets[dets > 0]
            
            if len(nonzero_dets) > 0:
                unique, counts_arr = np.unique(nonzero_dets, return_counts=True)
                for d, c in zip(unique, counts_arr):
                    d_int = int(d)
                    counts[d_int] = counts.get(d_int, 0) + int(c)
        
        return counts

    def calculate_max_score(points):
        counts = calculate_area_counts(points)
        if not counts:
            return 0
        return max(counts.values())

    def calculate_score_target(points, target_area):
        pts = np.array(points, dtype=np.int32)
        n = len(pts)
        count = 0
        p = pts
        
        for i in range(n - 2):
            dx = p[i+1:, 0] - p[i, 0]
            dy = p[i+1:, 1] - p[i, 1]
            
            det_matrix = np.abs(dx[:, None] * dy[None, :] - dy[:, None] * dx[None, :])
            
            rows, cols = np.triu_indices(n - i - 1, k=1)
            dets = det_matrix[rows, cols]
            
            count += int(np.sum(dets == target_area))
        
        return count

    # Try multiple initializations
    inits = [
        initial_grid_5x6(),
        initial_grid_6x5(),
        initial_grid_5x6_shifted(),
        initial_grid_6x5_shifted(),
        initial_lattice(),
    ]
    
    best_points = None
    best_score = 0
    best_area = None
    
    for init in inits:
        if time.time() - start_time > time_limit - 5:
            break
        score = calculate_max_score(init)
        if score > best_score:
            best_score = score
            best_points = init[:]
            counts = calculate_area_counts(init)
            best_area = max(counts, key=counts.get)
    
    # Simulated annealing
    current_points = best_points[:]
    current_score = best_score
    current_area = best_area
    
    def try_move(points, idx, new_x, new_y):
        """Try moving point idx to (new_x, new_y), return new points or None if invalid"""
        if -30 > new_x or new_x > 30 or -30 > new_y or new_y > 30:
            return None
        for j in range(30):
            if j != idx and points[j] == (new_x, new_y):
                return None
        new_points = points[:]
        new_points[idx] = (new_x, new_y)
        return new_points

    def sa_search(points, score, target_area, time_budget):
        """Simulated annealing search"""
        t_start = time.time()
        t_end = t_start + time_budget
        T = 5.0
        T_min = 0.1
        cooling = 0.9995
        
        best_local = points[:]
        best_local_score = score
        best_local_area = target_area
        
        while time.time() < t_end:
            # Choose a move
            idx = random.randint(0, 29)
            x, y = points[idx]
            
            # Random step
            step_size = random.choice([1, 1, 1, 2, 2, 3])
            dx = random.randint(-step_size, step_size)
            dy = random.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue
            
            new_x = x + dx
            new_y = y + dy
            
            new_points = try_move(points, idx, new_x, new_y)
            if new_points is None:
                continue
            
            new_score = calculate_max_score(new_points)
            
            # Determine if this is an improvement
            delta = new_score - score
            
            if delta > 0:
                # Accept improvement
                points = new_points
                score = new_score
                counts = calculate_area_counts(points)
                target_area = max(counts, key=counts.get)
                
                if score > best_local_score:
                    best_local_score = score
                    best_local = points[:]
                    best_local_area = target_area
            else:
                # Accept with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    points = new_points
                    score = new_score
            
            # Cool down
            T = max(T_min, T * cooling)
            
            # Periodically re-evaluate best area
            if random.random() < 0.01:
                counts = calculate_area_counts(points)
                if counts:
                    target_area = max(counts, key=counts.get)
        
        return best_local, best_local_score, best_local_area

    # Run SA with different time budgets
    remaining = time_limit - (time.time() - start_time)
    
    if remaining > 10:
        budget1 = min(30, remaining * 0.4)
        p1, s1, a1 = sa_search(current_points, current_score, current_area, budget1)
        if s1 > best_score:
            best_score = s1
            best_points = p1
            best_area = a1
    
    # Targeted polish: try to increase the dominant area count
    def targeted_polish(points, target_area, time_budget):
        """Try single-point moves to increase count of target_area"""
        t_start = time.time()
        t_end = t_start + time_budget
        current = points[:]
        current_count = calculate_score_target(current, target_area)
        best = current[:]
        best_count = current_count
        
        while time.time() < t_end:
            idx = random.randint(0, 29)
            x, y = current[idx]
            
            # Try moves in a wider range
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    new_pts = try_move(current, idx, new_x, new_y)
                    if new_pts is None:
                        continue
                    
                    new_count = calculate_score_target(new_pts, target_area)
                    if new_count > current_count:
                        current = new_pts
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best = current[:]
                        break
                else:
                    continue
                break
            
            # Also try random moves
            if random.random() < 0.3:
                idx = random.randint(0, 29)
                x, y = current[idx]
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                new_x = x + dx
                new_y = y + dy
                new_pts = try_move(current, idx, new_x, new_y)
                if new_pts is not None:
                    new_count = calculate_score_target(new_pts, target_area)
                    if new_count > current_count:
                        current = new_pts
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best = current[:]
        
        return best, best_count

    # Run targeted polish on best area
    remaining = time_limit - (time.time() - start_time)
    if remaining > 5 and best_area is not None:
        budget2 = min(20, remaining * 0.5)
        p2, c2 = targeted_polish(best_points, best_area, budget2)
        if c2 > best_score:
            # Check if overall max is also improved
            overall = calculate_max_score(p2)
            if overall > best_score:
                best_score = overall
                best_points = p2
            else:
                # The target area count improved but overall max might be different
                # Still keep if it's close
                pass
    
    # Final full polish
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        def final_polish(points, time_budget):
            t_start = time.time()
            t_end = t_start + time_budget
            current = points[:]
            current_score = calculate_max_score(current)
            best = current[:]
            best_score_local = current_score
            
            while time.time() < t_end:
                improved = False
                for idx in range(30):
                    x, y = current[idx]
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            new_x = x + dx
                            new_y = y + dy
                            new_pts = try_move(current, idx, new_x, new_y)
                            if new_pts is None:
                                continue
                            new_score = calculate_max_score(new_pts)
                            if new_score > current_score:
                                current = new_pts
                                current_score = new_score
                                improved = True
                                if current_score > best_score_local:
                                    best_score_local = current_score
                                    best = current[:]
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    # Try a random perturbation
                    idx = random.randint(0, 29)
                    x, y = current[idx]
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x + dx
                    new_y = y + dy
                    new_pts = try_move(current, idx, new_x, new_y)
                    if new_pts is not None:
                        current = new_pts
                        current_score = calculate_max_score(current)
            
            return best, best_score_local
        
        budget3 = min(10, remaining - 2)
        p3, s3 = final_polish(best_points, budget3)
        if s3 > best_score:
            best_score = s3
            best_points = p3
    
    return best_points
