import random
import numpy as np
from collections import Counter
import time

random.seed(42)

def construct():
    """
    Construct 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area (absolute determinant).
    """
    
    def get_doubled_area(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        
    def compute_max_multiplicity(points):
        n = len(points)
        areas = Counter()
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    da = get_doubled_area(points[i], points[j], points[k])
                    if da > 0:
                        areas[da] += 1
        if not areas:
            return 0, 0
        max_mult = max(areas.values())
        best_area = [a for a, m in areas.items() if m == max_mult][0]
        return max_mult, best_area

    def update_areas(points, idx, old_point, new_point, area_counts):
        n = len(points)
        # Remove old contributions
        for i in range(n):
            if i == idx: continue
            for j in range(i+1, n):
                if j == idx: continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = old_point
                da = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if da > 0:
                    area_counts[da] -= 1
                    if area_counts[da] == 0:
                        del area_counts[da]
                        
        # Add new contributions
        for i in range(n):
            if i == idx: continue
            for j in range(i+1, n):
                if j == idx: continue
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = new_point
                da = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if da > 0:
                    area_counts[da] += 1
                    
        return area_counts

    # Initial candidates
    candidates = []
    
    # 1. 5x6 Grid
    grid_5_6 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(grid_5_6)
    
    # 2. 6x5 Grid
    grid_6_5 = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(grid_6_5)
    
    # 3. 3x10 Grid
    grid_3_10 = [(x, y) for x in range(3) for y in range(10)]
    candidates.append(grid_3_10)
    
    # 4. 10x3 Grid
    grid_10_3 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(grid_10_3)
    
    # 5. 2x15 Grid
    grid_2_15 = [(x, y) for x in range(2) for y in range(15)]
    candidates.append(grid_2_15)
    
    # 6. 15x2 Grid
    grid_15_2 = [(x, y) for x in range(15) for y in range(2)]
    candidates.append(grid_15_2)
    
    best_points = None
    best_score = -1
    best_area_val = 0
    
    # Evaluate initial candidates
    for pts in candidates:
        score, area = compute_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts[:]
            best_area_val = area
            
    # Local Search
    current_points = best_points[:]
    n = len(current_points)
    
    # Pre-compute area counts for current_points
    current_areas = Counter()
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                da = get_doubled_area(current_points[i], current_points[j], current_points[k])
                if da > 0:
                    current_areas[da] += 1
    
    def get_max_mult(counter):
        if not counter: return 0
        return max(counter.values())
        
    current_max_mult = get_max_mult(current_areas)
    
    start_time = time.time()
    max_time = 55.0
    
    # To speed up, we maintain a set of occupied points for O(1) lookup
    occupied = set(current_points)
    
    while time.time() - start_time < max_time:
        idx = random.randint(0, n-1)
        old_point = current_points[idx]
        x_old, y_old = old_point
        
        # Try nearby moves first
        found_improvement = False
        # Define neighborhood
        # We'll try a 5x5 box around the old point
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                new_point = (x_old + dx, y_old + dy)
                if new_point in occupied:
                    continue
                if not (-30 <= new_point[0] <= 30 and -30 <= new_point[1] <= 30):
                    continue
                
                # Update areas
                # We need to copy the counter to test
                test_areas = current_areas.copy()
                update_areas(current_points, idx, old_point, new_point, test_areas)
                
                new_max_mult = get_max_mult(test_areas)
                
                if new_max_mult > current_max_mult:
                    # Accept
                    current_points[idx] = new_point
                    occupied.remove(old_point)
                    occupied.add(new_point)
                    current_areas = test_areas
                    current_max_mult = new_max_mult
                    best_score = current_max_mult
                    best_points = current_points[:]
                    found_improvement = True
                    break
            if found_improvement:
                break
                
        if not found_improvement:
            # Try a random global move occasionally
            if random.random() < 0.1:
                new_point = (random.randint(-30, 30), random.randint(-30, 30))
                if new_point in occupied:
                    continue
                test_areas = current_areas.copy()
                update_areas(current_points, idx, old_point, new_point, test_areas)
                new_max_mult = get_max_mult(test_areas)
                if new_max_mult > current_max_mult:
                    current_points[idx] = new_point
                    occupied.remove(old_point)
                    occupied.add(new_point)
                    current_areas = test_areas
                    current_max_mult = new_max_mult
                    best_score = current_max_mult
                    best_points = current_points[:]

    return list(best_points)
