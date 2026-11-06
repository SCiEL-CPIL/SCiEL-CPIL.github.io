import random
import time
import math
from itertools import combinations

def count_lines(points):
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = math.gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
            C = -dy * x1 + dx * y1
            key = (dx, dy, C)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    num_4 = 0
    num_5_plus = 0
    for key, pts in lines.items():
        if len(pts) == 4:
            num_4 += 1
        elif len(pts) >= 5:
            num_5_plus += 1
    return num_4, num_5_plus

def is_valid(points):
    if len(set(map(tuple, points))) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, num_5_plus = count_lines(points)
    if num_5_plus > 0:
        return False
    return True

def score(points):
    if not is_valid(points):
        return -1000
    num_4, _ = count_lines(points)
    return num_4

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # AG(2,4) has 20 lines of size 4.
    # Try to find integer coordinates that realize this.
    # Use local search with AG(2,4)-aware moves.
    
    # Initialize with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = score(best_points)
    
    # Multi-start local search
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            current = [p[:] for p in points]
        else:
            # Perturb from best or random
            if random.random() < 0.5:
                current = [p[:] for p in best_points]
                for _ in range(4):
                    idx = random.randint(0, 15)
                    current[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
            else:
                current = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(16)]
        
        # Ensure distinct
        for attempt in range(100):
            if len(set(map(tuple, current))) == 16:
                break
            idx = random.randint(0, 15)
            current[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
        
        current_score = score(current)
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current]
        
        # Local search
        no_improve = 0
        for step in range(200):
            if time.time() > deadline - 5:
                break
            if no_improve > 30:
                break
            
            idx = random.randint(0, 15)
            old_x, old_y = current[idx]
            
            # Try several moves
            improved = False
            for _ in range(10):
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                new_point = [new_x, new_y]
                if new_point in current:
                    continue
                
                current[idx] = new_point
                new_score = score(current)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return best_points
