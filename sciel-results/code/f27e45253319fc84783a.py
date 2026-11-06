import random
import math
import time

def get_line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def evaluate(points):
    n = len(points)
    if len(set(map(tuple, points))) != n:
        return False, 0
        
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            key = get_line_key(points[i], points[j])
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
            
    for key, pts in line_counts.items():
        if len(pts) >= 5:
            return False, 0
            
    score = 0
    for key, pts in line_counts.items():
        if len(pts) == 4:
            score += 1
            
    return True, score

def construct():
    start_time = time.time()
    limit = 110
    
    best_score = 0
    best_points = None
    
    # Multi-start with different initializations
    # 1. 4x4 Grid
    # 2. Random small box
    # 3. Random medium box
    
    while time.time() - start_time < limit:
        # Choose initialization strategy
        strat = random.randint(0, 2)
        if strat == 0:
            points = [[x, y] for x in range(4) for y in range(4)]
        elif strat == 1:
            points = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(16)]
        else:
            points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(16)]
            
        is_valid, current_score = evaluate(points)
        if not is_valid:
            # Try to repair or restart
            # For simplicity, just restart if invalid initially
            # But 4x4 is always valid. Random might not be.
            # If random is invalid, we can try to fix by moving points
            # Or just skip this restart if time is short
            if time.time() - start_time > limit - 5:
                break
            continue
            
        best_local_score = current_score
        best_local_points = [p[:] for p in points]
        
        # Local Search (SA)
        T = 5.0
        cooling_rate = 0.999
        
        while time.time() - start_time < limit - 10: # Leave time for next restart
            if random.random() < 0.5:
                idx = random.randint(0, 15)
                old_x, old_y = points[idx]
                new_x = old_x + random.randint(-5, 5)
                new_y = old_y + random.randint(-5, 5)
                
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                points[idx] = [new_x, new_y]
                
                is_valid, new_score = evaluate(points)
                
                if is_valid:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_points = [p[:] for p in points]
                    else:
                        points[idx] = [old_x, old_y]
                else:
                    points[idx] = [old_x, old_y]
            else:
                i, j = random.sample(range(16), 2)
                points[i], points[j] = points[j], points[i]
                
                is_valid, new_score = evaluate(points)
                
                if is_valid:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current_score = new_score
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_points = [p[:] for p in points]
                    else:
                        points[i], points[j] = points[j], points[i]
                else:
                    points[i], points[j] = points[j], points[i]
                    
            T *= cooling_rate
            if T < 0.1:
                T = 0.1
                
        if best_local_score > best_score:
            best_score = best_local_score
            best_points = best_local_points
            
        # If we found a very good score, we might stop early?
        # Upper bound is 15. If we hit 14 or 15, we can stop.
        if best_score >= 14:
            break

    return best_points
