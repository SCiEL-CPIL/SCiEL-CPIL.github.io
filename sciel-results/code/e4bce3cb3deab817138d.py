import time
import random
import math

def _count_lines_exactly_4(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns -1 if any line contains 5 or more points (invalid).
    Returns 0 if any line contains 5+ points? No, the task says "no five collinear".
    So if 5 are collinear, it's invalid.
    """
    n = len(points)
    # Check for duplicates
    if len(set(tuple(p) for p in points)) != n:
        return -1
        
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
            
    # Check collinearity and count 4-point lines
    # To avoid counting the same line multiple times, we can use a set of line identifiers.
    # A line can be identified by (A, B, C) in normalized form, or just by the set of points on it?
    # Simpler: Iterate over all pairs (i, j). Find the line. Count points on it.
    # To count each line once, we can store the frozenset of indices on the line in a set.
    
    lines_found = set()
    max_collinear = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # Normalize direction vector to identify the line uniquely?
            # Actually, we just need to know if a third point k is on this line.
            # We can check all k.
            
            # To avoid re-processing the same line, we can check if this line (defined by i,j) 
            # has been processed. But different pairs on the same line will generate the same line.
            # We can store the set of indices involved in the line.
            
            # Let's just count points on the line defined by i and j.
            count = 0
            indices_on_line = set()
            for k in range(n):
                pk = points[k]
                # Cross product to check collinearity
                # (p2 - p1) x (pk - p1) == 0
                # dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0
                if dx * (pk[1] - p1[1]) - dy * (pk[0] - p1[0]) == 0:
                    count += 1
                    indices_on_line.add(k)
            
            if count >= 5:
                return -1
                
            if count == 4:
                # Store a unique identifier for this line to avoid double counting
                # The sorted tuple of indices is a good ID
                line_id = tuple(sorted(indices_on_line))
                lines_found.add(line_id)
                
    return len(lines_found)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for 120s limit
    
    # Initial configuration: 4x4 grid
    # Centered to allow moves in both directions
    initial_points = [[x - 2, y - 2] for x in range(4) for y in range(4)]
    
    best_points = [p[:] for p in initial_points]
    best_score = _count_lines_exactly_4(best_points)
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Simple Hill Climbing with Random Restarts
    # We will try to improve current_score.
    # If stuck for a while, we restart from best or random.
    
    stuck_counter = 0
    max_stuck = 1000
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Decide step size
        if random.random() < 0.1:
            # Large jump
            new_x = old_x + random.randint(-20, 20)
            new_y = old_y + random.randint(-20, 20)
        else:
            # Small step
            new_x = old_x + random.randint(-3, 3)
            new_y = old_y + random.randint(-3, 3)
        
        # Ensure bounds
        new_x = max(-1000, min(1000, new_x))
        new_y = max(-1000, min(1000, new_y))
        
        # Avoid staying in place
        if new_x == old_x and new_y == old_y:
            continue
            
        current_points[idx] = [new_x, new_y]
        new_score = _count_lines_exactly_4(current_points)
        
        if new_score > current_score:
            current_score = new_score
            stuck_counter = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            stuck_counter += 1
            
            if stuck_counter > max_stuck:
                # Restart from best
                current_points = [p[:] for p in best_points]
                current_score = best_score
                stuck_counter = 0
                
                # Or try a random valid configuration?
                # For now, just restart from best.
                
    return best_points
