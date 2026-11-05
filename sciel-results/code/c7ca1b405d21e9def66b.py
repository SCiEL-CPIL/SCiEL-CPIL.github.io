import random
import time
import math
from collections import defaultdict

def construct():
    # Deadline guard: 110s to be safe against 150s kill
    start_time = time.time()
    deadline = start_time + 110
    
    def score(points):
        """
        Returns the number of lines containing exactly 4 points.
        Returns -1 if any 5 points are collinear or if points are not distinct.
        """
        n = len(points)
        if n != 16:
            return -1
        
        # Check distinct and bounds quickly
        seen = set()
        for p in points:
            if abs(p[0]) > 1000 or abs(p[1]) > 1000:
                return -1
            if p in seen:
                return -1
            seen.add(p)
            
        lines = defaultdict(int)
        
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                
                A = y1 - y2
                B = x2 - x1
                C = x1*y2 - x2*y1
                
                # Normalize
                # Optimization: Only compute GCD if not already 1?
                # math.gcd is fast in C, but call overhead exists.
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                # Ensure consistent sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                lines[(A, B, C)] += 1
                
        count_4 = 0
        for cnt in lines.values():
            if cnt >= 5:
                return -1
            if cnt == 4:
                count_4 += 1
                
        return count_4

    # Initial configuration: 4x4 grid (known to have 10 four-point lines)
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = score(best_points)
    
    # Local Search with Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 1.0
    cooling_rate = 0.99
    
    # We will run until deadline or a fixed number of iterations if it's fast
    iterations = 0
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Random perturbation
        if random.random() < 0.1:
            dx = random.randint(-50, 50)
            dy = random.randint(-50, 50)
        else:
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_point = [new_x, new_y]
        
        # Quick distinctness check
        is_distinct = True
        for k in range(16):
            if k == idx:
                continue
            if current_points[k] == new_point:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        current_points[idx] = new_point
        new_score = score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            if new_score == -1:
                current_points[idx] = [old_x, old_y]
            else:
                delta = new_score - current_score
                if random.random() < math.exp(delta / T):
                    current_score = new_score
                else:
                    current_points[idx] = [old_x, old_y]
                    
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        iterations += 1
        # Hard cap on iterations to prevent infinite loop if time check is slow?
        # 110 seconds / ~0.0001s per iter = 1.1M iters. 
        # Let's add a break if we've done too many.
        if iterations > 500000:
            break

    return best_points
