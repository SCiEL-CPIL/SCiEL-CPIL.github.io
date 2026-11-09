import time
import random
from itertools import combinations

def _count_exact4(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns -1 if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n != 16:
        return -1
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return -1
        
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
            
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line equation: Ax + By + C = 0
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            A = y1 - y2
            B = x2 - x1
            C = x1*y2 - x2*y1
            
            # Normalize (A, B, C) to ensure unique key for the same line
            # Find GCD of A, B, C
            g = 0
            for val in (A, B, C):
                g = __import__('math').gcd(g, val)
            if g == 0:
                continue # Should not happen for distinct points
            A //= g
            B //= g
            C //= g
            
            # Ensure a canonical sign (e.g., first non-zero is positive)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
                
            key = (A, B, C)
            
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    exact4_count = 0
    for key, pts_on_line in lines.items():
        if len(pts_on_line) >= 5:
            return -1 # Invalid: 5 collinear
        if len(pts_on_line) == 4:
            exact4_count += 1
            
    return exact4_count

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s buffer
    
    # Initial configuration: 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_exact4(best_points)
    
    # Local Search
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # To make the search more effective, we can try moving points to 
    # positions that are "likely" to form lines.
    # But random search with a good initial guess is a safe bet.
    
    # We will try a few hundred thousand steps if time permits.
    # Each step: pick a random point, pick a random new location, evaluate.
    
    # Optimization: Pre-compute valid moves? No, too many.
    # Just random sampling.
    
    # To avoid wasting time on obviously bad moves, we can restrict
    # the search space to a smaller box, e.g., -10 to 10.
    # The grid is 0..3. Expanding to -10..10 allows for new lines.
    
    search_box = 10
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0:
            # Optional: print or log, but we don't have stdout.
            pass
            
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate new point
        # Bias towards near the current point or near other points?
        # Let's try random in a box around the current point first, 
        # then random in a larger box.
        if random.random() < 0.5:
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            new_x = random.randint(-search_box, search_box)
            new_y = random.randint(-search_box, search_box)
            
        # Ensure distinct
        new_point = [new_x, new_y]
        if any(new_point == p for p in current_points):
            continue
            
        # Apply move
        current_points[idx] = new_point
        
        # Evaluate
        score = _count_exact4(current_points)
        
        if score > current_score:
            current_score = score
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in current_points]
        else:
            # Revert if not better (Hill Climbing)
            # To allow escaping local minima, we could accept worse moves with low prob,
            # but for this task, finding a global optimum (or close) is key.
            # Given the high baseline, hill climbing is safer.
            current_points[idx] = [old_x, old_y]
            
        # If we found a score > 10, we might want to try to refine it further
        # with smaller steps.
        
    return best_points

# EVOLVE-BLOCK-START
# (The code above is the content for the block)
# EVOLVE-BLOCK-END
