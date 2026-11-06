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
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: 3x4 grid
    # This is a known local minimum but not global.
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in initial_points]
    best_score = count_ordinary_lines(best_points)
    
    if is_all_collinear(best_points):
        best_points[0][1] += 1
        best_score = count_ordinary_lines(best_points)

    # Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    
    # We will run multiple restarts if time permits
    # But SA is stochastic, so we just run it for the duration
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_point = current_points[idx][:]
        
        # Generate a new point
        # Mix of small steps and larger jumps to escape basins
        if random.random() < 0.7:
            # Small step
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
        else:
            # Larger jump
            dx = random.randint(-20, 20)
            dy = random.randint(-20, 20)
            
        new_point = [old_point[0] + dx, old_point[1] + dy]
        
        # Check bounds
        if abs(new_point[0]) > 10**9 or abs(new_point[1]) > 10**9:
            continue
        
        # Check distinctness
        if new_point in current_points:
            continue
        
        # Swap
        current_points[idx] = new_point
        
        # Check collinearity
        if is_all_collinear(current_points):
            current_points[idx] = old_point
            continue
        
        new_score = count_ordinary_lines(current_points)
        
        # Metropolis criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                # Cool down slightly when improving
                T *= 0.99
        else:
            # Revert
            current_points[idx] = old_point
        
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            # Optional: Restart from best if stuck at low T
            if random.random() < 0.1:
                current_points = [p[:] for p in best_points]
                current_score = best_score
                T = 5.0  # Reheat

    # Final safety check
    if is_all_collinear(best_points):
        best_points[0][1] += 1
        while len(set(map(tuple, best_points))) < 12:
            best_points[0][0] += 1

    return best_points
