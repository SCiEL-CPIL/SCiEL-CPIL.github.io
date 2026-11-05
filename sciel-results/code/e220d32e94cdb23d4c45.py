import random
import math
from collections import defaultdict
import time

def construct():
    # Start with a 4x4 grid
    base = [-2, -1, 0, 1]
    points = [[x, y] for x in base for y in base]
    
    best_points = points[:]
    best_score = -1
    
    def evaluate(pts):
        line_counts = defaultdict(int)
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
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
                
                line_counts[(A, B, C)] += 1
        
        score = 0
        for key, pairs in line_counts.items():
            if pairs == 6:
                score += 1
            elif pairs > 6:
                return -1, False
        
        return score, True

    cur_score, cur_valid = evaluate(points)
    if cur_valid:
        best_score = cur_score
        best_points = points[:]
    else:
        best_score = 0
        best_points = points[:]

    start_time = time.time()
    timeout = 110
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # To escape local maxima, we use larger moves and allow degradation
    while time.time() - start_time < timeout and T > min_T:
        T *= cooling
        
        # Pick a random point
        idx = random.randint(0, 15)
        old_x, old_y = points[idx]
        
        # Generate a new position
        # Mix of small moves and large jumps
        if random.random() < 0.1:
            # Large jump: random point in a larger box
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
        else:
            # Small move
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx == 0 and dy == 0:
                continue
            new_x = old_x + dx
            new_y = old_y + dy
        
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        is_dup = False
        for i in range(16):
            if i != idx and points[i][0] == new_x and points[i][1] == new_y:
                is_dup = True
                break
        if is_dup:
            continue
            
        points[idx] = [new_x, new_y]
        
        new_score, new_valid = evaluate(points)
        
        if new_valid:
            delta = new_score - cur_score
            # Accept if better or with probability exp(delta/T) if worse
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = points[:]
            else:
                points[idx] = [old_x, old_y]
        else:
            points[idx] = [old_x, old_y]
            
    return best_points
