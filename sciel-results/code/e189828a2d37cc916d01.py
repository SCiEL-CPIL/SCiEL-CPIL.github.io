import time
from math import gcd
from itertools import combinations
import random

def count_ordinary_lines(points):
    """
    Count the number of lines containing exactly 2 of the given points.
    Optimized for speed using a dictionary of line signatures.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0 and dy == 0:
                continue
            
            # Normalize direction vector (dx, dy) to primitive form
            # Ensure unique representation: dx > 0 or (dx == 0 and dy > 0)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            g = gcd(abs(dx), abs(dy))
            if g > 1:
                dx //= g
                dy //= g
            
            # Line equation: dy * x - dx * y = c
            # c is an invariant for the line given the normalized direction
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            
            if key in line_counts:
                line_counts[key] += 1
            else:
                line_counts[key] = 1
    
    ordinary_count = 0
    for count in line_counts.values():
        if count == 1:
            ordinary_count += 1
            
    return ordinary_count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Deterministic Construction: Pappus Configuration Extension
    # The Pappus configuration is a (9_4) configuration.
    # We can embed it in the integer plane.
    # Standard Pappus points in projective plane, mapped to affine:
    # Lines L1: x=0, L2: x=1, L3: y=0, L4: y=1, L5: x+y=1, L6: x+y=2?
    # Actually, let's use the standard algebraic construction for 12 points.
    
    # A known optimal or near-optimal configuration for 12 points is derived from
    # the Hesse configuration or similar.
    # Let's try a specific set of 12 points that forms a dense arrangement.
    
    # Configuration A: 3x4 Grid
    grid_3x4 = [[x, y] for x in range(3) for y in range(4)]
    
    # Configuration B: Pappus-like 9 points + 3 more
    # Pappus points:
    # (0,0), (0,1), (0,2) on x=0
    # (1,0), (1,1), (1,2) on x=1
    # (2,0), (2,1), (2,2) on x=2
    # This is just a 3x3 grid.
    
    # Let's try a configuration based on the "Orchard Problem" optimal solutions.
    # For n=12, the minimum number of ordinary lines is 6.
    # One such configuration is:
    # 4 points on x=0: (0,0), (0,1), (0,2), (0,3)
    # 4 points on x=1: (1,0), (1,1), (1,2), (1,3)
    # 4 points on x=2: (2,0), (2,1), (2,2), (2,3)
    # This is a 3x4 grid.
    
    # Let's evaluate the 3x4 grid.
    best_points = grid_3x4
    best_score = count_ordinary_lines(best_points)
    
    # Let's try other small grids and structures.
    candidates = [
        [[x, y] for x in range(4) for y in range(3)], # 4x3
        [[x, y] for x in range(2) for y in range(6)], # 2x6
        [[x, y] for x in range(6) for y in range(2)], # 6x2
        [[x, y] for x in range(3) for y in range(4)], # 3x4
    ]
    
    for cand in candidates:
        score = count_ordinary_lines(cand)
        if score < best_score:
            best_score = score
            best_points = cand
    
    # If we haven't reached the theoretical minimum of 6, we use Simulated Annealing.
    # The key is to have a fast evaluation and good move generation.
    
    if best_score > 6:
        # Simulated Annealing
        current_points = [p[:] for p in best_points]
        current_score = best_score
        
        # Precompute some structures for faster delta evaluation?
        # For now, full recount is fast enough for 12 points (66 pairs).
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline and T > T_min:
            T *= cooling_rate
            
            # Generate a move: pick a random point and move it to a random nearby location
            i = random.randint(0, 11)
            old_x, old_y = current_points[i]
            
            # Move within a small box to maintain locality, or occasionally jump
            if random.random() < 0.1:
                # Large jump
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            else:
                # Small step
                new_x = old_x + random.randint(-2, 2)
                new_y = old_y + random.randint(-2, 2)
            
            # Check bounds
            if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                continue
                
            # Check distinctness
            new_point = (new_x, new_y)
            is_distinct = True
            for j in range(12):
                if j != i and (current_points[j][0], current_points[j][1]) == new_point:
                    is_distinct = False
                    break
                    
            if not is_distinct:
                continue
                
            # Apply move
            current_points[i] = [new_x, new_y]
            new_score = count_ordinary_lines(current_points)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Reject
                current_points[i] = [old_x, old_y]
                
            if best_score == 6:
                break

    return best_points
