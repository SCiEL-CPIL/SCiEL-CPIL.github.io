import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial configuration: 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    
    def count_ordinary_lines(pts):
        if len(pts) < 3:
            return 0
        
        lines = {}
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0:
                    continue
                
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return abs(a)
                
                g = gcd(gcd(A, B), C)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        ordinary_count = 0
        for key, indices in lines.items():
            if len(indices) == 2:
                ordinary_count += 1
        
        return ordinary_count

    def is_valid(pts):
        if len(set(tuple(p) for p in pts)) != 12:
            return False
        if len(pts) < 3:
            return True
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        all_collinear = True
        for i in range(2, len(pts)):
            x2, y2 = pts[i]
            if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) != 0:
                all_collinear = False
                break
        return not all_collinear

    best_points = points[:]
    best_score = count_ordinary_lines(best_points)
    
    # Local Search with Simulated Annealing-like acceptance
    current_score = best_score
    current_points = best_points[:]
    
    temp = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick a point
        idx = random.randint(0, 11)
        old_point = current_points[idx][:]
        
        # Move
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        new_point = [old_point[0] + dx, old_point[1] + dy]
        
        if abs(new_point[0]) > 10**9 or abs(new_point[1]) > 10**9:
            continue
            
        new_points = current_points[:]
        new_points[idx] = new_point
        
        if not is_valid(new_points):
            continue
            
        new_score = count_ordinary_lines(new_points)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            current_points = new_points
            if current_score < best_score:
                best_score = current_score
                best_points = current_points[:]
        else:
            # Accept worse with probability exp(-delta/T)
            delta = new_score - current_score
            if delta > 0:
                prob = random.random()
                if prob < 0.5 * (temp / 10.0): # Heuristic acceptance
                    current_score = new_score
                    current_points = new_points
        
        temp *= cooling_rate
        if temp < 0.1:
            temp = 0.1
            
    return best_points
