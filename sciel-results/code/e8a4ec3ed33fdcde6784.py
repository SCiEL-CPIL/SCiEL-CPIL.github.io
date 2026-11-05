import random
import math
from math import gcd

def count_ordinary(points):
    n = len(points)
    ordinary = 0
    lines_seen = set()
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a = dy
            b = -dx
            c = a*x1 + b*y1
            g2 = gcd(abs(a), gcd(abs(b), abs(c)))
            if g2 > 0:
                a //= g2
                b //= g2
                c //= g2
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            key = (a, b, c)
            if key in lines_seen:
                continue
            count = 0
            for k in range(n):
                xk, yk = points[k]
                if a*xk + b*yk == c:
                    count += 1
            lines_seen.add(key)
            if count == 2:
                ordinary += 1
    return ordinary

def construct():
    def simulated_annealing(points, iterations=5000, start_temp=100.0, end_temp=1.0):
        current_points = [p[:] for p in points]
        current_score = count_ordinary(current_points)
        best_points = [p[:] for p in current_points]
        best_score = current_score
        
        for i in range(iterations):
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = current_points[idx]
            
            # Random move: change x and y by random amounts
            # Use a range that allows exploration but keeps points within bounds
            # Since bounds are 10^9, we can move freely within a local area
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Ensure distinctness
            valid = True
            for k in range(12):
                if k == idx:
                    continue
                if current_points[k][0] == new_x and current_points[k][1] == new_y:
                    valid = False
                    break
            
            if not valid:
                continue
            
            current_points[idx] = [new_x, new_y]
            new_score = count_ordinary(current_points)
            
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = [old_x, old_y]
                
        return best_points, best_score

    best_global_points = None
    best_global_score = float('inf')
    
    # Multiple restarts
    for restart in range(50):
        # Initialize randomly within a small box to encourage structure
        # Points that are too far apart might not form good lines easily
        # A box of 20x20 is usually sufficient for 12 points to form interesting structures
        init_points = []
        for _ in range(12):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            init_points.append([x, y])
            
        # Ensure distinctness
        distinct_check = set(tuple(p) for p in init_points)
        if len(distinct_check) < 12:
            # Regenerate if collision
            continue
            
        res_points, res_score = simulated_annealing(init_points, iterations=2000)
        
        if res_score < best_global_score:
            best_global_score = res_score
            best_global_points = res_points
            
        if best_global_score <= 6:
            break
            
    return best_global_points
