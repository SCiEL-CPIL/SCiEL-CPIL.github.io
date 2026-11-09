import math
import time
import random
from math import gcd

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    
    line_pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    ordinary_count = 0
    for count in line_pair_counts.values():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_points = None
    best_count = float('inf')
    
    def validate_and_update(pts):
        nonlocal best_points, best_count
        if len(set(tuple(p) for p in pts)) != 12:
            return
        if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in pts):
            return
        c = count_ordinary_lines(pts)
        if c < best_count:
            best_count = c
            best_points = [list(p) for p in pts]
    
    # Try known good configurations
    # 3x4 grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    # 4x3 grid  
    grid_4x3 = [[x, y] for x in range(4) for y in range(3)]
    
    # Configuration with more collinearities:
    # Place points to maximize 3-point lines
    # Try: 4 lines with 3 points each, sharing some points
    # Example: (0,0),(1,0),(2,0) - horizontal
    # (0,0),(0,1),(0,2) - vertical  
    # (0,0),(1,1),(2,2) - diagonal
    # This gives 12 points with several 3-point lines
    
    config_collinear = [
        [0,0], [1,0], [2,0],   # line y=0
        [0,1], [0,2],          # line x=0 (shares (0,0))
        [1,1], [2,2],          # line y=x (shares (0,0))
        [3,0],                 # extra on y=0
        [0,3],                 # extra on x=0
        [1,2], [2,1],          # line x+y=3
        [3,3]                  # on y=x
    ]
    
    initial_configs = [grid_3x4, grid_4x3, config_collinear]
    
    random.seed(42)
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
        curr_pts = [list(p) for p in init_pts]
        validate_and_update(curr_pts)
        
        # Simulated Annealing with line snapping
        T = 2.0
        cooling_rate = 0.9995
        
        while time.time() < deadline:
            idx = random.randint(0, 11)
            old_x, old_y = curr_pts[idx]
            
            # Strategy 1: Line snapping (try to place on existing 2-point lines)
            if random.random() < 0.4:
                # Pick a random pair of other points and try to snap to their line
                others = [k for k in range(12) if k != idx]
                i, j = random.sample(others, 2)
                x1, y1 = curr_pts[i]
                x2, y2 = curr_pts[j]
                
                # Line through (x1,y1) and (x2,y2)
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx_n = dx // g
                dy_n = dy // g
                
                # Parametric: (x1 + t*dx_n, y1 + t*dy_n)
                # Try several t values
                placed = False
                for t in range(-5, 6):
                    if t == 0:
                        continue
                    nx = x1 + t * dx_n
                    ny = y1 + t * dy_n
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check distinctness
                    is_distinct = True
                    for k in range(12):
                        if k != idx and curr_pts[k] == [nx, ny]:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                    
                    curr_pts[idx] = [nx, ny]
                    new_count = count_ordinary_lines(curr_pts)
                    delta = new_count - count_ordinary_lines([list(p) for p in curr_pts[:idx]] + [[old_x, old_y]] + [list(p) for p in curr_pts[idx+1:]])
                    
                    if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                        placed = True
                        curr_pts[idx] = [nx, ny]
                        break
                    else:
                        curr_pts[idx] = [old_x, old_y]
                
                if placed:
                    validate_and_update(curr_pts)
                    T *= cooling_rate
                    if T < 0.01:
                        T = 0.01
                    continue
            
            # Strategy 2: Random move
            if random.random() < 0.7:
                new_x = old_x + random.randint(-3, 3)
                new_y = old_y + random.randint(-3, 3)
            else:
                new_x = old_x + random.randint(-15, 15)
                new_y = old_y + random.randint(-15, 15)
                
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            is_distinct = True
            for k in range(12):
                if k != idx and curr_pts[k] == [new_x, new_y]:
                    is_distinct = False
                    break
            if not is_distinct:
                continue
                
            curr_pts[idx] = [new_x, new_y]
            new_count = count_ordinary_lines(curr_pts)
            old_count = count_ordinary_lines([list(p) for p in curr_pts[:idx]] + [[old_x, old_y]] + [list(p) for p in curr_pts[idx+1:]])
            
            delta = new_count - old_count
            
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                validate_and_update(curr_pts)
            else:
                curr_pts[idx] = [old_x, old_y]
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
    return best_points
