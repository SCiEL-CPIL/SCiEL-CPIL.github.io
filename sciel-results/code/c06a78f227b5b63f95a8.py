import random
import time
import math
from math import gcd

def _get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = -(x2 - x1)
    c = x1 * y2 - x2 * y1
    
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        return None
    a //= g
    b //= g
    c //= g
    
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0:
        if b < 0:
            b, c = -b, -c
        elif b == 0:
            if c < 0:
                c = -c
    
    return (a, b, c)

def count_exact_4_lines(points):
    n = len(points)
    if n != 16:
        return 0
    if len(set(map(tuple, points))) != 16:
        return 0
    
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k > 4:
            return 0
        if k == 4:
            count += 1
            
    return count

def construct():
    start_time = time.time()
    time_limit = 115.0
    random.seed(42)
    
    # Generate diverse initial configurations
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted
    configs.append([[x+1, y+1] for x in range(4) for y in range(4)])
    
    # Points from a smaller grid with some perturbation
    base = [[x, y] for x in range(4) for y in range(4)]
    perturbed = [p[:] for p in base]
    for i in range(16):
        if random.random() < 0.3:
            perturbed[i] = [perturbed[i][0] + random.choice([-1, 1]), 
                            perturbed[i][1] + random.choice([-1, 1])]
    # Ensure distinct and in bounds
    seen = set()
    unique = []
    for p in perturbed:
        t = tuple(p)
        if t not in seen and abs(p[0]) <= 1000 and abs(p[1]) <= 1000:
            seen.add(t)
            unique.append(p)
    while len(unique) < 16:
        unique.append([random.randint(-10, 10), random.randint(-10, 10)])
        seen.add(tuple(unique[-1]))
    configs.append(unique[:16])
    
    # Two offset 4x2 rectangles
    pts = []
    for x in range(4):
        for y in range(2):
            pts.append([x*2, y*6])
            pts.append([x*2+1, y*6+3])
    configs.append(pts[:16])
    
    # Points on two parallel lines with offsets
    pts = []
    for i in range(8):
        pts.append([i*2, 0])
        pts.append([i*2+1, 5])
    configs.append(pts[:16])
    
    best_overall_score = 0
    best_overall_points = None
    
    for init_pts in configs:
        if time.time() - start_time > time_limit:
            break
        
        current_points = [p[:] for p in init_pts]
        current_score = count_exact_4_lines(current_points)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_points = [p[:] for p in current_points]
        
        if current_score <= 0:
            continue
        
        # Simulated annealing
        T = 1.5
        T_min = 0.01
        cooling = 0.997
        max_iter = 4000
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            if random.random() < 0.4:
                # Single point move
                idx = random.randint(0, 15)
                old_p = current_points[idx][:]
                # Try nearby moves first, then random
                if random.random() < 0.7:
                    dx = random.randint(-2, 2)
                    dy = random.randint(-2, 2)
                    new_x = old_p[0] + dx
                    new_y = old_p[1] + dy
                else:
                    new_x = random.randint(-15, 15)
                    new_y = random.randint(-15, 15)
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                current_points[idx] = [new_x, new_y]
                
                new_score = count_exact_4_lines(current_points)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
                    current_score = new_score
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_points = [p[:] for p in current_points]
                else:
                    current_points[idx] = old_p
            else:
                # Swap two points
                i, j = random.sample(range(16), 2)
                old_i = current_points[i][:]
                old_j = current_points[j][:]
                current_points[i] = old_j[:]
                current_points[j] = old_i[:]
                
                new_score = count_exact_4_lines(current_points)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < math.exp(delta / max(T, 0.001)):
                    current_score = new_score
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_overall_points = [p[:] for p in current_points]
                else:
                    current_points[i] = old_i
                    current_points[j] = old_j
                    
            T = max(T * cooling, T_min)
        
        # Restart with best found so far if we improved
        if best_overall_points and best_overall_score > current_score:
            current_points = [p[:] for p in best_overall_points]
            current_score = best_overall_score
    
    # Final validation
    if best_overall_points is None:
        fallback = [[x, y] for x in range(4) for y in range(4)]
        return fallback
    
    final_score = count_exact_4_lines(best_overall_points)
    if final_score <= 0:
        fallback = [[x, y] for x in range(4) for y in range(4)]
        if count_exact_4_lines(fallback) > 0:
            return fallback
        return [[i, 0] for i in range(16)]
    
    return best_overall_points
