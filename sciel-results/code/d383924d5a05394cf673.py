import time
import random
from itertools import combinations
from math import gcd

def count_exact_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    
    # For each pair of points, determine the line they define
    # Then count how many points lie on each line
    # Use a dictionary: line -> set of point indices on that line
    # Line representation: normalized (a, b, c) for ax + by + c = 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through (x1,y1) and (x2,y2): 
            # (y2-y1)x - (x2-x1)y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x2
            
            # Normalize the line equation
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            # Ensure consistent sign
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    # Count lines with exactly 4 points
    count = 0
    for line, pts in line_to_points.items():
        if len(pts) == 4:
            count += 1
    
    return count

def has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    for line, pts in line_to_points.items():
        if len(pts) >= 5:
            return True
    
    return False

def is_valid(points):
    """Check all constraints: distinct, bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    
    # Check distinct
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    
    # Check no 5 collinear
    if has_five_collinear(points):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    random.seed(42)
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = count_exact_four_lines(best_points)
    
    # Local search with simulated annealing
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Temperature for simulated annealing
    T = 1.0
    cooling = 0.9995
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Try several random new positions
        found_improvement = False
        for attempt in range(50):
            # Random new position
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
            
            # Check if this position is already taken
            if [new_x, new_y] in current_points:
                continue
            
            # Create new configuration
            new_points = [p[:] for p in current_points]
            new_points[idx] = [new_x, new_y]
            
            # Check validity
            if not is_valid(new_points):
                continue
            
            # Count exact-4 lines
            new_score = count_exact_four_lines(new_points)
            
            # Accept if better, or with some probability if worse (simulated annealing)
            delta = new_score - current_score
            if delta >= 0 or random.random() < pow(2.718281828, delta / T):
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_points = [p[:] for p in current_points]
                    best_score = current_score
                
                found_improvement = True
                break
        
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 0.01
        
        # Occasionally restart from best
        if iterations % 500 == 0 and random.random() < 0.1:
            current_points = [p[:] for p in best_points]
            current_score = best_score
            T = 0.5
    
    return best_points
