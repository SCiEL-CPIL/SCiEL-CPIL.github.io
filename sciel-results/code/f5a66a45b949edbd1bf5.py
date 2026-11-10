import time
import random
import math

def count_lines(points):
    n = len(points)
    lines = set()
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                continue
                
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            lines.add((A, B, C))
            
    num_4 = 0
    num_5 = 0
    
    for A, B, C in lines:
        count = 0
        for x, y in points:
            if A*x + B*y + C == 0:
                count += 1
        if count == 4:
            num_4 += 1
        elif count >= 5:
            num_5 += 1
            
    return num_4, num_5

def construct():
    start_time = time.time()
    timeout = 145
    
    best_score = -1
    best_points = None
    
    box = 50
    
    # Multiple restarts
    while time.time() - start_time < timeout:
        # Random initialization
        pts = []
        while len(pts) < 16:
            x = random.randint(-box, box)
            y = random.randint(-box, box)
            if [x, y] not in pts:
                pts.append([x, y])
                
        # Simulated Annealing
        T = 10.0
        current_pts = pts
        current_4, current_5 = count_lines(current_pts)
        current_score = current_4 - 1000 * current_5
        
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_pts]
            
        while T > 0.01 and time.time() - start_time < timeout:
            # Pick a random point to move
            i = random.randint(0, 15)
            old_pt = current_pts[i]
            
            # New random position
            new_x = random.randint(-box, box)
            new_y = random.randint(-box, box)
            
            # Check if point already exists
            conflict = False
            for k in range(16):
                if k != i and current_pts[k] == [new_x, new_y]:
                    conflict = True
                    break
            if conflict:
                continue
                
            current_pts[i] = [new_x, new_y]
            
            new_4, new_5 = count_lines(current_pts)
            new_score = new_4 - 1000 * new_5
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_pts]
            else:
                current_pts[i] = old_pt
                
            T *= 0.99 # Cooling rate
            
    return best_points
