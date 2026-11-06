import time
import random
import math

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _count_lines_exactly_4(points):
    n = len(points)
    if n < 4:
        return 0, True
    
    line_pairs = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - y2 * x1
            
            g = _gcd(_gcd(a, b), c)
            if g != 0:
                a //= g
                b //= g
                c //= g
            else:
                continue
                
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
                
            key = (a, b, c)
            line_pairs[key] = line_pairs.get(key, 0) + 1
            
    count_4 = 0
    is_valid = True
    
    for key, pair_count in line_pairs.items():
        val = 1 + 8 * pair_count
        root = int(math.isqrt(val))
        if root * root != val:
            continue
        k = (1 + root) // 2
        if k >= 5:
            is_valid = False
        elif k == 4:
            count_4 += 1
            
    return count_4, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _count_lines_exactly_4(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score, is_valid = _count_lines_exactly_4(current_points)
    
    if not is_valid:
        current_points = [p[:] for p in best_points]
        current_score = best_score

    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Choose operation: Move (60%) or Swap (40%)
        if random.random() < 0.6:
            idx = random.randint(0, 15)
            old_point = current_points[idx][:]
            
            if random.random() < 0.5:
                nx = old_point[0] + random.randint(-3, 3)
                ny = old_point[1] + random.randint(-3, 3)
            else:
                nx = random.randint(-1000, 1000)
                ny = random.randint(-1000, 1000)
                
            if nx < -1000 or nx > 1000 or ny < -1000 or ny > 1000:
                continue
            if [nx, ny] in current_points:
                continue
                
            current_points[idx] = [nx, ny]
            new_score, is_valid = _count_lines_exactly_4(current_points)
            
            if is_valid:
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = old_point
            else:
                current_points[idx] = old_point
                
        else:
            idx1 = random.randint(0, 15)
            idx2 = random.randint(0, 15)
            if idx1 == idx2:
                continue
                
            old1 = current_points[idx1][:]
            old2 = current_points[idx2][:]
            
            current_points[idx1] = old2
            current_points[idx2] = old1
            
            new_score, is_valid = _count_lines_exactly_4(current_points)
            
            if is_valid:
                if new_score >= current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = [p[:] for p in current_points]
                else:
                    current_points[idx1] = old1
                    current_points[idx2] = old2
            else:
                current_points[idx1] = old1
                current_points[idx2] = old2

        # Random restart
        if iterations % 20000 == 0 and random.random() < 0.05:
             current_points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
             unique = set()
             for p in current_points:
                 if p in unique:
                     p[0] = (p[0] + 1) % 2001 - 1000
                 unique.add(p)
             current_score, is_valid = _count_lines_exactly_4(current_points)
             if not is_valid:
                 current_points = [p[:] for p in best_points]
                 current_score = best_score

    return best_points
