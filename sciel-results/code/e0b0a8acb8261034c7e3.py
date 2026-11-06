import itertools
import random
import math

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Dictionary to store lines as (A, B, C) normalized
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y1 - y2
            B = x2 - x1
            C = -(A * x1 + B * y1)
            
            if A == 0 and B == 0:
                continue
            
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(gcd(A, B), C)
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            
            lines[key].add(i)
            lines[key].add(j)
            
    ordinary = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary += 1
            
    return ordinary

def construct():
    # Initial configuration: 3x4 grid
    points = []
    for r in range(3):
        for c in range(4):
            points.append([c, r])
            
    current_count = count_ordinary_lines(points)
    
    best_points = [p[:] for p in points]
    best_count = current_count
    
    # Simulated Annealing
    # Parameters
    start_temp = 100.0
    end_temp = 0.01
    cooling_rate = 0.995
    max_iterations = 5000
    move_range = 10
    
    current_points = [p[:] for p in points]
    current_count = count_ordinary_lines(current_points)
    
    temp = start_temp
    
    for _ in range(max_iterations):
        if temp <= end_temp:
            break
            
        # Generate a random move
        new_points = [p[:] for p in current_points]
        idx = random.randint(0, 11)
        old_x, old_y = new_points[idx]
        
        dx = random.randint(-move_range, move_range)
        dy = random.randint(-move_range, move_range)
        
        if dx == 0 and dy == 0:
            continue
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds (keep them reasonable to avoid huge numbers, though 10^9 is allowed)
        # We can scale up later if needed, but small integers are easier to handle
        if new_x < -50 or new_x > 50 or new_y < -50 or new_y > 50:
            continue
            
        # Check distinctness
        exists = False
        for k in range(12):
            if k != idx and new_points[k][0] == new_x and new_points[k][1] == new_y:
                exists = True
                break
        if exists:
            continue
            
        new_points[idx] = [new_x, new_y]
        
        new_count = count_ordinary_lines(new_points)
        
        # Acceptance criterion
        delta = new_count - current_count
        
        if delta < 0:
            # Better, accept
            current_points = new_points
            current_count = new_count
        else:
            # Worse, accept with probability exp(-delta / temp)
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                current_points = new_points
                current_count = new_count
                
        # Update best
        if current_count < best_count:
            best_count = current_count
            best_points = [p[:] for p in current_points]
            
        # Cool down
        temp *= cooling_rate
        
    return best_points
