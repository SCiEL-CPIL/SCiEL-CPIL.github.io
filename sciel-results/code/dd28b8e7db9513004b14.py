import random
import time
import math
from collections import defaultdict
from fractions import Fraction

def count_lines(points):
    """
    Counts the number of lines containing exactly 4 points.
    Returns (num_lines_4, is_valid).
    is_valid is False if any line has 5 or more points.
    """
    n = len(points)
    if n != 16:
        return 0, False
    
    line_counts = defaultdict(int)
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By = C
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize (A, B, C)
            # Find GCD of A, B, C
            # Handle zeros
            g = 0
            for val in (abs(A), abs(B), abs(C)):
                if val > 0:
                    g = math.gcd(g, val)
            if g == 0:
                continue # Should not happen for distinct points
            
            A //= g
            B //= g
            C //= g
            
            # Fix sign: first non-zero of A, B should be positive
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
                
            line_counts[(A, B, C)] += 1
            
    num_4_lines = 0
    is_valid = True
    
    for count in line_counts.values():
        if count == 4:
            num_4_lines += 1
        elif count >= 5:
            is_valid = False
            # We can break early if we just need validity, 
            # but we need the count of 4s too. 
            # If invalid, the score is effectively 0 or penalized.
            
    return num_4_lines, is_valid

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial configuration: 4x4 grid
    # Points: (0,0) to (3,3)
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = count_lines(best_points)
    
    # Also try a few other simple patterns?
    # Maybe a slightly sheared grid?
    # x' = x + y, y' = y
    # This is an affine transformation, so same score.
    
    # Local Search
    current_points = [p[:] for p in best_points]
    current_score, _ = count_lines(current_points)
    
    # Neighborhood radius
    radius = 5
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Generate new position
        # Try a few random candidates to find a valid one
        new_x = old_x + random.randint(-radius, radius)
        new_y = old_y + random.randint(-radius, radius)
        
        # Check bounds
        if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
            continue
            
        # Check distinctness
        is_distinct = True
        for k, p in enumerate(current_points):
            if k != idx and p[0] == new_x and p[1] == new_y:
                is_distinct = False
                break
        if not is_distinct:
            continue
            
        # Update and evaluate
        current_points[idx] = [new_x, new_y]
        new_score, is_valid = count_lines(current_points)
        
        if is_valid:
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            elif new_score == current_score:
                # Accept with some probability to escape local minima
                if random.random() < 0.1:
                    current_score = new_score
            else:
                # Reject worse scores
                # Revert
                current_points[idx] = [old_x, old_y]
        else:
            # Invalid configuration (5+ collinear)
            # Revert
            current_points[idx] = [old_x, old_y]
            
    return best_points

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the correct format.
# EVOLVE-BLOCK-END
