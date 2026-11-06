import random
import math
import time
from collections import defaultdict

def count_4_lines(points):
    """
    Returns the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check distinctness
    if len(set(map(tuple, points))) != 16:
        return -1
        
    # Check bounds
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1

    lines = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate line equation: A*x + B*y = C
            A = dy
            B = -dx
            C = A*x1 + B*y1
            
            # Normalize the line equation
            if A == 0 and B == 0:
                continue
            
            g_line = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g_line == 0:
                g_line = 1
            A_n = A // g_line
            B_n = B // g_line
            C_n = C // g_line
            
            # Ensure canonical form: A_n > 0 or (A_n == 0 and B_n > 0)
            if A_n < 0 or (A_n == 0 and B_n < 0):
                A_n = -A_n
                B_n = -B_n
                C_n = -C_n
                
            key = (A_n, B_n, C_n)
            lines[key].append(i)
            
    count_4 = 0
    for key, pts in lines.items():
        if len(pts) > 4:
            return -1
        if len(pts) == 4:
            count_4 += 1
            
    return count_4

def construct():
    # Better initial configuration: start with a perturbed grid
    # Use a 4x4 grid but with some random offsets to break 5-collinear
    
    random.seed(42)
    
    def generate_initial():
        # Start with 4x4 grid
        pts = []
        for i in range(4):
            for j in range(4):
                x = i * 3 + random.randint(-1, 1)
                y = j * 3 + random.randint(-1, 1)
                pts.append([x, y])
        return pts
    
    best_score = -1
    best_points = None
    
    # Try multiple initial configurations
    for trial in range(20):
        points = generate_initial()
        score = count_4_lines(points)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in points]
    
    if best_points is None:
        # Fallback to simple grid
        best_points = [[i, j] for i in range(4) for j in range(4)]
        best_score = count_4_lines(best_points)
    
    # Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < 110:
        # Random move: pick a point and move it
        i = random.randint(0, 15)
        
        # Small random step
        step_size = random.randint(1, 5)
        delta_x = random.randint(-step_size, step_size)
        delta_y = random.randint(-step_size, step_size)
        
        new_points = [p[:] for p in points]
        new_x = new_points[i][0] + delta_x
        new_y = new_points[i][1] + delta_y
        
        # Keep within bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
            
        new_points[i][0] = new_x
        new_points[i][1] = new_y
        
        new_score = count_4_lines(new_points)
        
        if new_score < 0:
            continue
            
        delta = new_score - count_4_lines(points)
        
        if delta > 0 or random.random() < math.exp(delta / T):
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                
        T *= cooling
    
    # Final verification
    if count_4_lines(best_points) < 0:
        # Fallback
        fallback = [[i, j] for i in range(4) for j in range(4)]
        if count_4_lines(fallback) >= 0:
            return fallback
        return best_points
    
    return best_points
