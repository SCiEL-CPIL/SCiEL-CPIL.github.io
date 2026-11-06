"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def count_4_lines(points):
    """Count lines with exactly 4 points, return (count, has_five_collinear)."""
    n = len(points)
    lines = set()
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        if g == 0:
            continue
        dx, dy = dx // g, dy // g
        # Canonical form: make (dx, dy) unique
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        # Count points on this line
        cnt = 0
        for k in range(n):
            xk, yk = points[k]
            if (xk - x1) * dy == (yk - y1) * dx:
                cnt += 1
        if cnt >= 4:
            # Create a canonical line key
            # Use the line equation: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            # Normalize sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy, c = -dx, -dy, -c
            lines.add((dx, dy, c))
    # Check for 5 collinear
    for i, j in combinations(range(n), 2):
        x1, y1 = points[i]
        x2, y2 = points[j]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        cnt = 0
        for k in range(n):
            xk, yk = points[k]
            if (xk - x1) * dy == (yk - y1) * dx:
                cnt += 1
        if cnt >= 5:
            return 0, True
    return len(lines), False

def initial_grid():
    """Create a 4x4 grid as starting point."""
    points = []
    for x in range(4):
        for y in range(4):
            points.append([x, y])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with the 4x4 grid (score 10)
    best_points = initial_grid()
    best_score, _ = count_4_lines(best_points)
    
    # Simulated annealing
    current_points = initial_grid()
    current_score, _ = count_4_lines(current_points)
    
    temp = 5.0
    cooling = 0.999
    min_temp = 0.01
    
    while temp > min_temp:
        if time.time() > deadline - 2.0:
            break
        
        # Try a random move: pick a point and move it
        idx = random.randint(0, 15)
        old_x, old_y = current_points[idx]
        
        # Move to a nearby random position
        new_x = old_x + random.randint(-3, 3)
        new_y = old_y + random.randint(-3, 3)
        
        # Check bounds
        if abs(new_x) > 1000 or abs(new_y) > 1000:
            continue
        
        # Check if point is already used
        new_pt = [new_x, new_y]
        if any(p[0] == new_x and p[1] == new_y for p in current_points if p is not current_points[idx]):
            continue
        
        # Apply move
        current_points[idx] = new_pt
        new_score, has_five = count_4_lines(current_points)
        
        if has_five:
            # Reject moves that create 5 collinear
            current_points[idx] = [old_x, old_y]
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
                current_points[idx] = [old_x, old_y]
        
        temp *= cooling
    
    # If we found something better, return it
    if best_score > 10:
        return best_points
    
    # Otherwise return the 4x4 grid
    return initial_grid()
# EVOLVE-BLOCK-END
