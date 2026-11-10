import time
import random
import math

def count_lines_fast(points):
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            if A == 0 and B == 0:
                continue
            
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C)) if C != 0 else math.gcd(abs(A), abs(B))
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
            
    four_lines = 0
    for count_pairs in line_counts.values():
        discriminant = 1 + 8 * count_pairs
        sqrt_disc = int(math.isqrt(discriminant))
        if sqrt_disc * sqrt_disc == discriminant:
            k = (1 + sqrt_disc) // 2
            if k == 4:
                four_lines += 1
            elif k >= 5:
                return -1
    return four_lines

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -1
    best_points = None
    
    # Candidate 1: 4x4 Grid
    grid_4x4 = [[x, y] for x in range(-2, 2) for y in range(-2, 2)]
    score = count_lines_fast(grid_4x4)
    if score > best_score:
        best_score = score
        best_points = [p[:] for p in grid_4x4]
        
    # Candidate 2: 5x5 Grid Subsets
    # Try all combinations of 16 points from 25? Too many (5M).
    # Try random subsets.
    grid_5x5 = [[x, y] for x in range(-2, 3) for y in range(-2, 3)]
    for _ in range(100):
        subset = random.sample(grid_5x5, 16)
        score = count_lines_fast(subset)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in subset]
            
    # Local Search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    while time.time() < deadline:
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Small step
        if random.random() < 0.8:
            step = random.randint(1, 2)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
        else:
            step = random.randint(3, 10)
            dx = random.randint(-step, step)
            dy = random.randint(-step, step)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_pt = [new_x, new_y]
        # Check distinct
        valid = True
        for i, p in enumerate(current_points):
            if i != idx and p == new_pt:
                valid = False
                break
        if not valid:
            continue
            
        current_points[idx] = new_pt
        new_score = count_lines_fast(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            current_points[idx] = [old_x, old_y]
            
    if best_score < 0:
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
