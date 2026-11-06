"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def count_4_lines(points):
    """
    Counts lines with exactly 4 points.
    Returns (count, has_five_collinear).
    """
    n = len(points)
    if n < 4:
        return 0, False
    
    line_pair_counts = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            # Normalize direction vector
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            
            # Canonical sign: dx > 0 or dx == 0 and dy > 0
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            
            # Line constant c: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            
            key = (dx, dy, c)
            line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
            
    count_4 = 0
    has_5 = False
    
    for cnt in line_pair_counts.values():
        # cnt is C(k, 2)
        if cnt == 6:
            count_4 += 1
        elif cnt >= 10:
            has_5 = True
            return 0, True
            
    return count_4, has_5

def initial_grid():
    """Create a 4x4 grid as starting point."""
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x, y])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Start with the 4x4 grid (score 10)
    best_points = initial_grid()
    best_score, _ = count_4_lines(best_points)
    
    # Simulated Annealing
    current_points = [p[:] for p in best_points]
    current_score, _ = count_4_lines(current_points)
    
    temp = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    # Pre-calculate valid random points for jumps to avoid bias or invalid coords
    # Actually, generating on the fly is fine if we check bounds.
    
    while temp > min_temp:
        if time.time() > deadline:
            break
        
        # Choose move type
        if random.random() < 0.7:
            # Local move
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        else:
            # Global jump
            idx = random.randint(0, 15)
            new_x = random.randint(-1000, 1000)
            new_y = random.randint(-1000, 1000)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check distinctness
        new_pt = [new_x, new_y]
        is_valid_distinct = True
        for k in range(16):
            if k != idx and current_points[k][0] == new_x and current_points[k][1] == new_y:
                is_valid_distinct = False
                break
        
        if not is_valid_distinct:
            continue
        
        # Apply move
        old_pt = current_points[idx][:]
        current_points[idx] = new_pt
        
        new_score, has_five = count_4_lines(current_points)
        
        if has_five:
            # Reject moves that create 5 collinear
            current_points[idx] = old_pt
            continue
        
        if new_score >= current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp(-delta/temp)
            delta = current_score - new_score
            if random.random() < pow(2.718281828, -delta / temp):
                current_score = new_score
            else:
                # Revert
                current_points[idx] = old_pt
        
        temp *= cooling
        
    return best_points
# EVOLVE-BLOCK-END
