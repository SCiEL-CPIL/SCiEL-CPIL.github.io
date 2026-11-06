import time
import random
from math import gcd

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    A = dy
    B = -dx
    C = -(A*x1 + B*y1)
    g2 = gcd(abs(A), gcd(abs(B), abs(C)))
    if g2 != 0:
        A //= g2
        B //= g2
        C //= g2
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_lines(points):
    n = len(points)
    line_pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key:
                line_pairs[key] = line_pairs.get(key, 0) + 1
    
    count_4 = 0
    count_5 = 0
    
    for key, pair_count in line_pairs.items():
        if pair_count >= 6:
            A, B, C = key
            cnt = 0
            for p in points:
                if A*p[0] + B*p[1] + C == 0:
                    cnt += 1
            if cnt == 4:
                count_4 += 1
            elif cnt >= 5:
                count_5 += 1
                
    return count_4, count_5

def fitness(points):
    if len(set(tuple(p) for p in points)) < 16:
        return -10**9
    c4, c5 = count_lines(points)
    return c4 - 1000 * c5

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_points = None
    best_fit = -10**9
    
    # Try multiple independent SA runs
    for run in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        points = [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]
        
        T = 50.0
        current_fit = fitness(points)
        no_improve_count = 0
        local_best_fit = current_fit
        local_best_points = [p[:] for p in points]
        
        while time.time() < deadline - 5:
            p_new = [p[:] for p in points]
            idx = random.randint(0, 15)
            move_type = random.random()
            
            if move_type < 0.6:
                # Nudge
                axis = random.randint(0, 1)
                delta = random.randint(-5, 5)
                if delta != 0:
                    p_new[idx][axis] += delta
                    if p_new[idx][axis] > 1000: p_new[idx][axis] = 1000
                    if p_new[idx][axis] < -1000: p_new[idx][axis] = -1000
            elif move_type < 0.8:
                # Medium Jump
                axis = random.randint(0, 1)
                p_new[idx][axis] = random.randint(-1000, 1000)
            else:
                # Large Jump (both coords)
                p_new[idx][0] = random.randint(-1000, 1000)
                p_new[idx][1] = random.randint(-1000, 1000)
            
            new_fit = fitness(p_new)
            
            if new_fit > current_fit:
                points = p_new
                current_fit = new_fit
                no_improve_count = 0
                if current_fit > local_best_fit:
                    local_best_fit = current_fit
                    local_best_points = [p[:] for p in points]
            else:
                if random.random() < pow(2.71828, (new_fit - current_fit) / T):
                    points = p_new
                    current_fit = new_fit
                no_improve_count += 1
            
            T *= 0.9995
            
            if no_improve_count > 1000:
                # Reset
                T = 50.0
                no_improve_count = 0
                # Optionally restart from local best
                points = [p[:] for p in local_best_points]
                current_fit = local_best_fit
                
            if time.time() > deadline - 5:
                break
        
        if local_best_fit > best_fit:
            best_fit = local_best_fit
            best_points = [p[:] for p in local_best_points]
            
    # Verify best_points validity
    if best_points:
        c4, c5 = count_lines(best_points)
        if c5 == 0 and len(set(tuple(p) for p in best_points)) == 16:
            return best_points
    
    # Fallback
    return [[x, y] for x in range(4) for y in range(4)]
