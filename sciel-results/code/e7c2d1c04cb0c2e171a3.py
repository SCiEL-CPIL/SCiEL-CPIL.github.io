import random
import time
import math

def _count_ordinary_lines(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_pairs = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Line equation: A*x + B*y + C = 0
            A = dy
            B = -dx
            C = -(A * x1 + B * y1)
            
            # Normalize (A, B, C)
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                # This shouldn't happen for distinct points
                g = 1
            A //= g
            B //= g
            C //= g
            
            # Canonical form
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_pairs[key] = line_pairs.get(key, 0) + 1
            
    ordinary_count = 0
    for count in line_pairs.values():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def _local_search(points, max_iters, deadline):
    best = [p[:] for p in points]
    best_score = _count_ordinary_lines(best)
    
    if best_score == 6:
        return best, best_score
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        new_points = [p[:] for p in best]
        idx = random.randint(0, len(best) - 1)
        
        # Try moving to a random location
        # Range can be adjusted, but small integers are often sufficient for these configs
        new_points[idx] = [random.randint(-50, 50), random.randint(-50, 50)]
        
        # Ensure distinctness
        is_distinct = True
        for k in range(len(new_points)):
            if k == idx:
                continue
            if new_points[k] == new_points[idx]:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        score = _count_ordinary_lines(new_points)
        
        if score < best_score:
            best = new_points
            best_score = score
            if best_score == 6:
                return best, best_score
                
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Initial configurations
    configs = []
    
    # 3x4 Grid
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    configs.append(grid_3x4)
    
    # 3x3 Grid + 3 random
    grid_3x3 = [[x, y] for x in range(3) for y in range(3)]
    for _ in range(5):
        extra = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(3)]
        # Ensure no duplicates with grid
        valid_extra = []
        for p in extra:
            if p not in grid_3x3 and p not in valid_extra:
                valid_extra.append(p)
        if len(valid_extra) == 3:
            configs.append(grid_3x3 + valid_extra)
            
    # Random
    for _ in range(5):
        random_config = [[random.randint(-20, 20), random.randint(-20, 20)] for _ in range(12)]
        # Check distinct
        if len(set(map(tuple, random_config))) == 12:
            configs.append(random_config)

    best_overall = None
    best_score_overall = 100
    
    for config in configs:
        if time.time() > deadline:
            break
        res, score = _local_search(config, 2000, deadline)
        if score < best_score_overall:
            best_overall = res
            best_score_overall = score
            if best_score_overall == 6:
                break

    if best_overall is None:
        return grid_3x4
        
    return best_overall
