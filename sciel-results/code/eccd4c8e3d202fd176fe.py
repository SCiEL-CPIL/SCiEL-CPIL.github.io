import time
import random
from math import gcd

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    # Use a dictionary to count pairs per line
    # Key: normalized line equation (A, B, C) for Ax + By + C = 0
    line_pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize the line equation
            # Find GCD of A, B, C
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure a consistent sign
            # A > 0 or (A == 0 and B > 0) or (A == 0 and B == 0 and C > 0)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    # A line with k points has C(k, 2) pairs.
    # Ordinary lines have exactly 2 points, so C(2, 2) = 1 pair.
    # We just need to count how many lines have exactly 1 pair.
    ordinary_count = 0
    for count in line_pair_counts.values():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_count = float('inf')
    
    # Initial configurations
    # 1. 4x3 Grid
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    
    # 2. 3x4 Grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    
    # 3. Random initial points
    def random_points():
        pts = set()
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            pts.add((x, y))
        return list(pts)

    initial_configs = [grid_4x3, grid_3x4, random_points(), random_points()]
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
            
        # Verify distinctness and bounds
        if len(set(tuple(p) for p in init_pts)) != 12:
            continue
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in init_pts):
            continue
            
        curr_pts = [list(p) for p in init_pts]
        curr_count = count_ordinary_lines(curr_pts)
        
        if curr_count < best_count:
            best_count = curr_count
            best_points = [list(p) for p in curr_pts]
            
        # Local Search: Hill Climbing
        no_improvement = 0
        max_no_improvement = 5000
        temperature = 1.0
        
        while time.time() < deadline and no_improvement < max_no_improvement:
            # Pick a random point to move
            idx = random.randint(0, 11)
            old_x, old_y = curr_pts[idx]
            
            # Generate a new position
            # Bias towards nearby points to allow gradual improvement
            # Or random jump
            if random.random() < 0.5:
                # Small move
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
            else:
                # Large move
                new_x = random.randint(-20, 20)
                new_y = random.randint(-20, 20)
            
            # Ensure distinctness
            if any(curr_pts[i][0] == new_x and curr_pts[i][1] == new_y for i in range(12) if i != idx):
                no_improvement += 1
                continue
                
            # Ensure bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                no_improvement += 1
                continue
                
            # Evaluate new configuration
            test_pts = curr_pts.copy()
            test_pts[idx] = [new_x, new_y]
            new_count = count_ordinary_lines(test_pts)
            
            if new_count < curr_count:
                curr_pts = test_pts
                curr_count = new_count
                no_improvement = 0
                if curr_count < best_count:
                    best_count = curr_count
                    best_points = [list(p) for p in curr_pts]
                    if best_count == 6:
                        return best_points
            else:
                # Accept worse moves with some probability (Simulated Annealing style)
                # to escape local minima
                if random.random() < 0.05:
                    curr_pts = test_pts
                    curr_count = new_count
                no_improvement += 1
                
            # Reset no_improvement counter occasionally to allow exploration
            if no_improvement > 1000:
                # Random restart within current local search? Or just keep going
                pass

    # If no good configuration found, return the best one (likely grid)
    if best_points is None:
        best_points = grid_4x3
        
    return best_points
