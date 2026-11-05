import random
import time
from math import gcd

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    points: list of [x, y]
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line key to count of points on that line
    # Line key: (A, B, C) for Ax + By + C = 0, normalized
    line_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C
            # (y2 - y1)x - (x2 - x1)y + (x2*y1 - x1*y2) = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize
            if A == 0 and B == 0:
                continue # Duplicate points, should not happen
            
            g = gcd(gcd(abs(A), abs(B)), abs(C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure a consistent sign (e.g., first non-zero is positive)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            
            if key not in line_points:
                line_points[key] = 1
            else:
                line_points[key] += 1
                
    # A line is ordinary if it contains exactly 2 points.
    # In our map, a line with k points has k*(k-1)/2 pairs.
    # Wait, the map counts PAIRS.
    # If a line has 2 points, it has 1 pair.
    # If a line has 3 points, it has 3 pairs.
    # So we can't just check if count == 1 in the pair map.
    # We need to reconstruct point counts or store sets.
    
    # Let's restart the counting logic to store sets of points or just counts of points.
    # Better: Store the set of points for each line? Memory heavy but n=12 is small.
    # Or, simply:
    # For each pair, find line.
    # We need to know how many points are on that line.
    # We can use a dictionary: line_key -> set of point indices.
    
    line_indices = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            if A == 0 and B == 0:
                continue
            
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
            
            if key not in line_indices:
                line_indices[key] = set()
            
            line_indices[key].add(i)
            line_indices[key].add(j)
            
    ordinary_count = 0
    for key, indices in line_indices.items():
        if len(indices) == 2:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x3 grid
    # This has 24 ordinary lines.
    initial_points = [[x, y] for x in range(4) for y in range(3)]
    
    best_points = [p[:] for p in initial_points]
    best_score = count_ordinary_lines(best_points)
    
    current_points = [p[:] for p in initial_points]
    current_score = best_score
    
    # Local search
    # We will try to move points to reduce the score.
    # Random restarts if stuck.
    
    attempts = 0
    max_attempts = 100000 # Will be limited by time
    
    while time.time() < deadline:
        attempts += 1
        
        # Pick a random point to move
        idx = random.randint(0, 11)
        old_x, old_y = current_points[idx]
        
        # Generate a new position
        # Try small moves first, then larger
        # Strategy: try to move to a position that lies on an existing line with >= 2 other points
        # This is complex. Simple random walk is okay for small n.
        
        # Let's try a random offset
        if random.random() < 0.5:
            # Small step
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        else:
            # Large step
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
        new_x = old_x + dx
        new_y = old_y + dy
        
        # Check bounds
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        # Check distinctness
        valid = True
        for i in range(12):
            if i == idx:
                continue
            if current_points[i][0] == new_x and current_points[i][1] == new_y:
                valid = False
                break
                
        if not valid:
            continue
            
        # Move and evaluate
        current_points[idx] = [new_x, new_y]
        new_score = count_ordinary_lines(current_points)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                # Optional: restart if we hit a very good score
                if best_score <= 6:
                    break
        else:
            # Revert
            current_points[idx] = [old_x, old_y]
            
        # Occasionally restart to escape local minima
        if attempts % 5000 == 0:
            # Try a different initial configuration
            # e.g. shifted grid, or random
            if random.random() < 0.5:
                # Shifted grid
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                current_points = [[x + offset_x, y + offset_y] for x in range(4) for y in range(3)]
            else:
                # Random points in a small box
                current_points = [[random.randint(-10, 10), random.randint(-10, 10)] for _ in range(12)]
                # Ensure distinct
                # Simple check, if duplicates, just regenerate
                while len(set(map(tuple, current_points))) < 12:
                    i = random.randint(0, 11)
                    current_points[i] = [random.randint(-10, 10), random.randint(-10, 10)]
                    
            current_score = count_ordinary_lines(current_points)
            if current_score < best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
                
    return best_points
