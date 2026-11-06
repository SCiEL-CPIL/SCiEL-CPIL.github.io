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
    
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return False
    
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x2 - x1) * (yi - y1) != (y2 - y1) * (xi - x1):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin for 150s limit
    
    # Try several structured initial configurations
    # Configuration 1: 4x3 grid
    configs = []
    
    # 4x3 grid: 12 points
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Near-pencil: 11 on x-axis, 1 off
    configs.append([[i, 0] for i in range(11)] + [[0, 1]])
    
    # 2 lines with 6 points each (sharing 2 points): 6+6-2=10... need 12
    # Line 1: y=0, x=0..5 (6 points)
    # Line 2: y=1, x=0..5 (6 points) - but these are parallel, no shared points
    # Actually let's do: line y=0 with 7 points, line y=1 with 5 points
    configs.append([[i, 0] for i in range(7)] + [[i, 1] for i in range(5)])
    
    # 3 lines: each with 4 points, forming a triangle with points at intersections
    # Line 1: y=0, x=0,1,2,3
    # Line 2: y=1, x=0,1,2,3  
    # Line 3: y=2, x=0,1,2,3
    # That's 12 points, 3 lines with 4 each
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Better: use a 3x3 grid (9 points) + 3 more points strategically
    # 3x3 grid at (0,0) to (2,2), plus 3 points at infinity approximated by far points
    # Actually, let's try: 3x3 grid + 3 points on a new line
    base = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 collinear points not on any existing line through 2 grid points
    # Points (5,5), (6,6), (7,7) - line y=x, but (0,0),(1,1),(2,2) are on y=x
    # So use (5,0), (6,0), (7,0) - line y=0, but (0,0),(1,0),(2,0) are on y=0
    # Use (5,1), (6,1), (7,1) - line y=1, but (0,1),(1,1),(2,1) are on y=1
    # Use (5,3), (6,3), (7,3) - line y=3, no grid points on this line
    configs.append(base + [[5, 3], [6, 3], [7, 3]])
    
    # Try: 4 points on y=0, 4 on y=1, 4 on y=2, but shifted
    # This is just a 4x3 grid
    
    # Try a different structure: 6 points on one line, 6 on another, sharing 2 points
    # Line 1: (0,0), (1,0), (2,0), (3,0), (4,0), (5,0)
    # Line 2: (0,1), (1,1), (2,1), (3,1), (4,1), (5,1) - parallel, no shared
    # Make them intersect: Line 1: y=0, Line 2: y=x
    # Line 1 points: (0,0), (1,0), (2,0), (3,0), (4,0), (5,0)
    # Line 2 points: (0,0), (1,1), (2,2), (3,3), (4,4), (5,5)
    # Shared: (0,0). Total: 6+6-1=11. Need one more.
    # Add (6,0) to line 1: now 7 on line 1, 6 on line 2, shared (0,0)
    # Total: 7+6-1=12
    configs.append([(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), 
                    (1,1), (2,2), (3,3), (4,4), (5,5)])
    
    # Try: 4 lines with 3 points each, forming a complete quadrilateral
    # This is getting complex. Let's just evaluate all configs and pick best.
    
    best_points = None
    best_score = float('inf')
    
    for config in configs:
        # Convert tuples to lists
        pts = [list(p) for p in config]
        # Check distinctness
        if len(set(map(tuple, pts))) != 12:
            continue
        # Check bounds
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            continue
        if is_all_collinear(pts):
            continue
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        best_points = [[i, 0] for i in range(11)] + [[0, 1]]
        best_score = count_ordinary_lines(best_points)
    
    # Local search: try moving individual points to reduce ordinary lines
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            orig_x, orig_y = current_points[i]
            # Try small perturbations
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    new_point = [new_x, new_y]
                    # Check distinctness
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
    
    # If we found something with score 6, that's optimal
    if best_score <= 6:
        return best_points
    
    # Try simulated annealing for more exploration
    T = 5.0
    cooling = 0.99
    min_T = 0.1
    
    while time.time() < deadline and T > min_T:
        i = random.randint(0, 11)
        orig_x, orig_y = current_points[i]
        
        # Random jump
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        
        new_x = orig_x + dx
        new_y = orig_y + dy
        
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            T *= cooling
            continue
            
        new_point = [new_x, new_y]
        
        is_distinct = True
        for j in range(12):
            if j == i:
                continue
            if current_points[j] == new_point:
                is_distinct = False
                break
        
        if not is_distinct:
            T *= cooling
            continue
            
        old_point = current_points[i]
        current_points[i] = new_point
        
        if is_all_collinear(current_points):
            current_points[i] = old_point
            T *= cooling
            continue
            
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    current_points[i] = old_point
        
        T *= cooling
    
    return best_points
