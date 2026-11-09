import math
import time
import random
from math import gcd
from collections import defaultdict

def count_ordinary(points):
    n = len(points)
    if n < 3:
        return 0
    
    # Identify unique lines
    lines = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - y2 * x1
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                g = 1
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            lines.add((A, B, C))
            
    # Count points on each line
    ordinary_count = 0
    for A, B, C in lines:
        count = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                count += 1
        if count == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    # Known good configurations to start with
    # The theoretical minimum for 12 points is 6 ordinary lines.
    # A 3x4 grid has 6 ordinary lines. Let's verify and use it as a starting point.
    # Also try other structured configurations.
    
    configs = []
    
    # 3x4 Grid (rows=3, cols=4)
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 Grid (rows=4, cols=3)
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Pappus-like or other structured sets?
    # Let's try a 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 6x2 Grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 1x12 line (bad, 11 ordinary lines)
    # Skip this one as it's bad.
    
    # A configuration with points on a conic?
    # Parabola y=x^2
    configs.append([[x, x*x] for x in range(12)])
    
    best_points = None
    best_score = float('inf')
    
    # First, evaluate all initial configs
    for init_points in configs:
        # Ensure distinct
        seen = set()
        unique_points = []
        for p in init_points:
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_points.append(p)
        
        # If not enough points, fill with random
        while len(unique_points) < 12:
            p = [random.randint(-10, 10), random.randint(-10, 10)]
            if tuple(p) not in seen:
                seen.add(tuple(p))
                unique_points.append(p)
        
        current_points = [p[:] for p in unique_points]
        current_score = count_ordinary(current_points)
        
        if current_score < best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
    
    # If we already found 6, we can stop early as it's the theoretical minimum
    if best_score == 6:
        return best_points
    
    # Simulated Annealing
    # We will run SA from the best found configuration
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0
    cooling_rate = 0.9995
    
    # Precompute line counts for faster updates? 
    # For now, full recount is O(n^2 * n) = O(n^3) which is 12^3 = 1728 operations.
    # 150s / 1728 ~ 86k steps. That's not enough for good SA.
    # We need a faster counting method or incremental updates.
    
    # Let's implement a faster counting method using a dictionary of lines.
    # Actually, the current count_ordinary is O(n^3).
    # n=12, pairs=66. For each pair, we compute the line. Then for each line, we count points.
    # Number of unique lines is at most 66.
    # So it's 66 * 12 = 792 operations. Plus the 66 gcd computations.
    # This is fast.
    
    # Let's optimize the inner loop of SA.
    
    while time.time() < deadline:
        # Pick a random point to move
        i = random.randint(0, 11)
        orig_p = current_points[i][:]
        
        # Generate new position
        # Try different step sizes
        step_size = random.choice([1, 2, 5, 10])
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        new_x = orig_p[0] + dx
        new_y = orig_p[1] + dy
        
        # Check bounds and distinctness
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
        if [new_x, new_y] in current_points:
            continue
            
        current_points[i] = [new_x, new_y]
        new_score = count_ordinary(current_points)
        
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                if best_score == 6:
                    return best_points
        else:
            current_points[i] = orig_p
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            # Restart with higher temperature
            T = 5.0
            # Randomize current state slightly? Or just keep going.
            # Let's restart from best_points to escape local minima
            current_points = [p[:] for p in best_points]
            current_score = best_score
            
    return best_points
