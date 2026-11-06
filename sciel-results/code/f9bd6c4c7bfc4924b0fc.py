import time
import random
import math

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly 2 points from the given list.
    points: list of [x, y]
    """
    n = len(points)
    if n < 3:
        return 0
    
    lines = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line equation: A*x + B*y = C
            # A = y1 - y2
            # B = x2 - x1
            # C = A*x1 + B*y1
            
            A = y1 - y2
            B = x2 - x1
            C = A * x1 + B * y1
            
            # Normalize the line representation
            # Find GCD of A, B, C to normalize
            # Handle case where A or B is 0
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Ensure a consistent sign (e.g., first non-zero among A, B is positive)
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_key = (A, B, C)
            
            if line_key not in lines:
                lines[line_key] = set()
            
            lines[line_key].add(i)
            lines[line_key].add(j)
            
    ordinary_count = 0
    for key, pts_in_line in lines.items():
        if len(pts_in_line) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    # Start with a 3x4 grid
    # Scale it up to allow for moves
    scale = 1000
    points = [[x * scale, y * scale] for x in range(4) for y in range(3)]
    
    current_ordinary = count_ordinary_lines(points)
    best_points = points.copy()
    best_ordinary = current_ordinary
    
    start_time = time.time()
    time_limit = 110  # Leave some margin for 120s limit
    
    random.seed(42)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        i = random.randint(0, 11)
        old_x, old_y = points[i]
        
        # Generate a new position
        # We can try a random jump or a small step
        # Since we want to find integer coordinates, let's try random coordinates in a range
        # Or better, try to snap to a grid or random nearby
        
        # Let's try a random nearby point
        # Radius can be adaptive
        radius = 500
        new_x = old_x + random.randint(-radius, radius)
        new_y = old_y + random.randint(-radius, radius)
        
        # Ensure within bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check if point is distinct from others
        is_distinct = True
        for k in range(12):
            if k == i:
                continue
            if points[k][0] == new_x and points[k][1] == new_y:
                is_distinct = False
                break
        
        if not is_distinct:
            continue
            
        # Update point
        points[i] = [new_x, new_y]
        
        # Count ordinary lines
        new_ordinary = count_ordinary_lines(points)
        
        if new_ordinary < current_ordinary:
            current_ordinary = new_ordinary
            if current_ordinary < best_ordinary:
                best_ordinary = current_ordinary
                best_points = [p[:] for p in points]
        else:
            # Revert if not better (simple hill climbing)
            # Or accept with probability p if equal
            points[i] = [old_x, old_y]
            
    return best_points
