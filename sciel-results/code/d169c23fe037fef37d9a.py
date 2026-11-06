import time
import random
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly two of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            g = gcd(gcd(A, B), C)
            if g == 0:
                continue
            A //= g
            B //= g
            C //= g
            
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            lines[key] = lines.get(key, 0) + 1
    
    ordinary_count = 0
    for key, count in lines.items():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def is_all_collinear(points):
    if len(points) < 3:
        return False
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    
    # Check if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1)
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return False
    
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit
    
    # Initial configuration: 4x3 grid
    # This has 24 ordinary lines.
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in initial_points]
    best_score = count_ordinary_lines(best_points)
    
    # If the initial configuration is all collinear (it's not), handle it.
    if is_all_collinear(best_points):
         # Fallback to a known non-collinear set
         best_points = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,0], [3,1], [3,2]]
         best_score = count_ordinary_lines(best_points)

    # Local Search with larger steps and explicit collinearity check
    for restart in range(50):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_points = [p[:] for p in initial_points]
        else:
            # Random perturbation from best
            current_points = [p[:] for p in best_points]
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, 11)
                # Larger jumps to escape local minima
                current_points[idx][0] += random.randint(-10, 10)
                current_points[idx][1] += random.randint(-10, 10)
                
                # Ensure distinctness
                if len(set(map(tuple, current_points))) < 12:
                    # Revert if collision
                    current_points[idx][0] -= random.randint(-10, 10)
                    current_points[idx][1] -= random.randint(-10, 10)
                    # If still collision, try again or just accept a valid random restart
                    if len(set(map(tuple, current_points))) < 12:
                         current_points[idx][0] = random.randint(-20, 20)
                         current_points[idx][1] = random.randint(-20, 20)
                         if len(set(map(tuple, current_points))) < 12:
                             current_points[idx][0] = 100
                             current_points[idx][1] = 100

        # Check validity
        if is_all_collinear(current_points):
            # If all collinear, force one point off
            current_points[0][1] += 1
            if len(set(map(tuple, current_points))) < 12:
                 current_points[0][0] += 1

        current_score = count_ordinary_lines(current_points)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try moving each point
            for i in range(12):
                if time.time() > deadline:
                    break
                orig_x, orig_y = current_points[i]
                
                # Try nearby coordinates with varying step sizes
                # Mix of small and large steps
                steps = [-1, 1, -2, 2, -5, 5, -10, 10]
                for dx in steps:
                    for dy in steps:
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        
                        # Check bounds
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        
                        # Check distinctness
                        new_point = [new_x, new_y]
                        is_distinct = True
                        for j in range(12):
                            if j == i:
                                continue
                            if current_points[j] == new_point:
                                is_distinct = False
                                break
                        
                        if not is_distinct:
                            continue
                        
                        old_point = current_points[i]
                        current_points[i] = new_point
                        
                        # CRITICAL CHECK: Not all collinear
                        if is_all_collinear(current_points):
                            current_points[i] = old_point
                            continue
                        
                        new_score = count_ordinary_lines(current_points)
                        
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_points = [p[:] for p in current_points]
                        else:
                            current_points[i] = old_point
        
        if time.time() > deadline:
            break
            
    # Final safety check
    if is_all_collinear(best_points):
        # This should not happen, but just in case
        best_points[0][1] += 1
        while len(set(map(tuple, best_points))) < 12:
             best_points[0][0] += 1

    return best_points

# EVOLVE-BLOCK-END
