import time
import random
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
def _count_4_lines(points):
    """
    Counts the number of lines containing exactly 4 of the given points.
    Also returns True if any line contains 5 or more points (invalid).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    # Map line representation -> count of points on it
    line_counts = defaultdict(int)
    
    # Normalize line representation: (A, B, C) where Ax + By + C = 0
    # To ensure uniqueness, we normalize (A, B, C) such that:
    # 1. gcd(A, B, C) = 1
    # 2. A > 0, or (A == 0 and B > 0), or (A == 0 and B == 0 and C > 0)
    # Note: Since points are integers, A, B, C can be integers.
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            # Line through (x1, y1) and (x2, y2)
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            
            # Normalize
            if A == 0 and B == 0:
                continue # Should not happen for distinct points
                
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g > 0:
                A //= g
                B //= g
                C //= g
            
            # Sign normalization
            if A < 0:
                A = -A
                B = -B
                C = -C
            elif A == 0:
                if B < 0:
                    B = -B
                    C = -C
                elif B == 0:
                    if C < 0:
                        C = -C
            
            line_counts[(A, B, C)] += 1

    count_4 = 0
    has_5 = False
    for count in line_counts.values():
        if count == 4:
            count_4 += 1
        elif count >= 5:
            has_5 = True
            
    return count_4, has_5

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 150s limit, keep 35s safety margin
    
    # Initial configuration: 4x4 grid
    # This is known to be valid (no 5 collinear) and has some 4-lines.
    # Grid points: (0,0) to (3,3)
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Calculate initial score
    current_score, is_invalid = _count_4_lines(initial_points)
    
    best_points = [p[:] for p in initial_points]
    best_score = current_score
    
    # Local Search / Simulated Annealing
    # We will randomly pick a point and move it to a nearby integer coordinate.
    # We track line counts incrementally? No, full recount is O(N^2) = 256 ops.
    # 16 points -> 120 pairs. Recounting is very fast.
    # Let's estimate: 120 pairs * ~10 ops = 1200 ops per eval.
    # 115 seconds / 1200 ops is plenty for millions of steps if optimized in C, 
    # but in Python we need to be careful. 
    # Let's try to do as many steps as possible.
    
    random.seed(42)
    
    # Define a neighborhood
    # Move a point to (x+dx, y+dy) where dx, dy in {-2, -1, 0, 1, 2}
    # Ensure new point is distinct from others and within bounds |x|,|y| <= 1000
    
    current_points = [p[:] for p in initial_points]
    current_score_val = current_score
    
    # To speed up, we can precompute something? 
    # Actually, let's just run the loop.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        ox, oy = current_points[idx]
        
        # Generate a random new position
        # Bias towards nearby points to maintain structure, but allow jumps
        # Let's try a mix: 50% small step, 50% larger step
        if random.random() < 0.5:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
        else:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            
        nx = ox + dx
        ny = oy + dy
        
        # Check bounds
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
            
        # Check distinctness
        # Is (nx, ny) already in current_points (excluding idx)?
        is_dup = False
        for k in range(16):
            if k == idx:
                continue
            if current_points[k][0] == nx and current_points[k][1] == ny:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Tentatively move
        current_points[idx] = [nx, ny]
        
        # Evaluate
        new_score, is_invalid = _count_4_lines(current_points)
        
        if is_invalid:
            # Reject invalid moves (5 collinear)
            # Unless we are in a very high temperature phase? 
            # For now, strict constraint: reject if invalid.
            # This might get stuck, but the 4x4 grid is a good basin.
            current_points[idx] = [ox, oy]
        else:
            if new_score > current_score_val:
                current_score_val = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = [p[:] for p in current_points]
            else:
                # Accept worse moves with some probability (annealing)
                # Simple random walk: accept with prob 0.1 to escape local minima
                if random.random() < 0.1:
                    current_score_val = new_score
                else:
                    # Revert
                    current_points[idx] = [ox, oy]
                    
        # Optional: If we find a significantly better score, reset random seed or something?
        # No, keep going.
        
    return best_points
# EVOLVE-BLOCK-END
