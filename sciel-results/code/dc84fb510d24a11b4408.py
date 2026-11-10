"""Minimize ordinary lines for 12 integer points via enhanced local search."""
import time
import random
import math
from itertools import combinations

DEADLINE = time.time() + 140  # safety margin before 150s kill

def _line_key(p1, p2):
    """Canonical key for line through p1=(x1,y1) and p2=(x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        return (0, 0, x1)  # vertical line: x = x1
    # Normalize: make dx > 0, reduce by gcd
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    if dx < 0:
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            line_counts[key] = line_counts.get(key, 0) + 1
    # A line with k points is counted C(k,2) times in the pairs.
    # Ordinary lines are those with exactly 2 points, i.e., counted once.
    ordinary = sum(1 for v in line_counts.values() if v == 1)
    return ordinary

def _is_valid(points):
    """Check: 12 distinct integer points, |x|,|y| <= 1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Check not all collinear
    if len(points) < 3:
        return True
    p0, p1, p2 = points[0], points[1], points[2]
    (x0, y0), (x1, y1), (x2, y2) = p0, p1, p2
    if (x1 - x0) * (y2 - y0) != (y1 - y0) * (x2 - x0):
        return True
    # All three collinear, check all
    for i in range(3, 12):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def construct():
    random.seed(42)
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = count_ordinary_lines(best_points)
    
    # Generate diverse initial seeds
    seeds = []
    
    # 1. Grids
    seeds.append([[x, y] for x in range(4) for y in range(3)])
    seeds.append([[x, y] for x in range(3) for y in range(4)])
    seeds.append([[x, y] for x in range(6) for y in range(2)])
    
    # 2. Points on few lines
    # 6 on x-axis, 6 on y=1
    seeds.append([[i, 0] for i in range(6)] + [[i, 1] for i in range(6)])
    # 4 on x-axis, 4 on y=1, 4 on y=2
    seeds.append([[i, 0] for i in range(4)] + [[i, 1] for i in range(4)] + [[i, 2] for i in range(4)])
    # 3 on each of 4 lines
    seeds.append([[i, 0] for i in range(3)] + [[i, 1] for i in range(3)] + 
                 [[i, 2] for i in range(3)] + [[i, 3] for i in range(3)])
    
    # 3. Regular polygon approximations (scaled to integers)
    # Hexagon
    seeds.append([[3, 0], [1, 2], [-1, 2], [-3, 0], [-1, -2], [1, -2],
                  [6, 0], [2, 4], [-2, 4], [-6, 0], [-2, -4], [2, -4]])
    
    # 4. Random seeds
    for s in range(50):
        random.seed(s)
        pts = []
        for _ in range(12):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            pts.append([x, y])
        if _is_valid(pts):
            seeds.append(pts)
    
    # Evaluate all seeds
    for s in seeds:
        if time.time() > DEADLINE - 5:
            break
        if not _is_valid(s):
            continue
        sc = count_ordinary_lines(s)
        if sc < best_score:
            best_score = sc
            best_points = [p[:] for p in s]
    
    # Local Search
    points = [p[:] for p in best_points]
    score = best_score
    
    # Adaptive step size
    step_size = 20
    
    # Main search loop
    for iteration in range(100000):
        if time.time() > DEADLINE - 5:
            break
        if score <= 6:
            break
            
        idx = random.randint(0, 11)
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            continue
            
        old_x, old_y = points[idx]
        new_x, new_y = old_x + dx, old_y + dy
        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
            continue
            
        new_pt = [new_x, new_y]
        valid = True
        for i in range(12):
            if i != idx and points[i] == new_pt:
                valid = False
                break
        if not valid:
            continue
            
        points[idx] = new_pt
        new_score = count_ordinary_lines(points)
        
        if new_score <= score:
            score = new_score
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
                # Reduce step size on improvement
                step_size = max(1, step_size // 2)
        else:
            points[idx] = [old_x, old_y]
            
        # Occasionally increase step size to escape local minima
        if iteration % 1000 == 0 and step_size < 100:
            step_size = min(100, step_size * 2)
            
    return best_points
