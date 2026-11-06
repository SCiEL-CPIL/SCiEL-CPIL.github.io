import time
import random
from math import gcd

def _count_ordinary(points):
    """Count ordinary lines (lines containing exactly 2 of the given points)."""
    n = len(points)
    if n < 2:
        return 0
    # For each pair, determine the line, count how many points are on it.
    # A line is ordinary iff it contains exactly 2 points.
    # Each ordinary line is counted exactly once (by its unique pair).
    ordinary = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2)
            # Check if any other point k is collinear
            is_ordinary = True
            for k in range(n):
                if k == i or k == j:
                    continue
                x3, y3 = points[k]
                # Cross product check
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    is_ordinary = False
                    break
            if is_ordinary:
                ordinary += 1
    return ordinary


def _is_valid(points):
    """Check that points are 12 distinct integer points, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        key = (x, y)
        if key in seen:
            return False
        seen.add(key)
    # Check not all collinear
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, 12):
        x3, y3 = points[i]
        if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
            return True
    return False


def _pappus_configuration():
    """
    Pappus configuration: 9 points, 9 lines, each line has 3 points.
    This gives 0 ordinary lines among the 9 points.
    We need 12 points. Add 3 more points carefully.
    
    Standard Pappus: Take two lines L1, L2. Place 3 points A,B,C on L1 and 3 points D,E,F on L2.
    The 9 intersection points of lines AD, BE, CF and AE, BD, CF... 
    Actually, the Pappus theorem says that for two lines with 3 points each,
    the 9 intersection points of cross-connections lie on 9 lines with 3 points each.
    
    Let me use a concrete Pappus configuration:
    L1: y=0, points A=(0,0), B=(1,0), C=(2,0)
    L2: y=1, points D=(0,1), E=(1,1), F=(2,1)
    
    Pappus lines:
    - AD and BE: A=(0,0),D=(0,1) -> x=0; B=(1,0),E=(1,1) -> x=1. They don't intersect in a new point in the standard sense...
    
    Let me use a different standard Pappus setup.
    Actually, the Pappus configuration has 9 points and 9 lines. Let me look up a concrete one.
    
    Standard Pappus:
    Points on line 1: A, B, C
    Points on line 2: D, E, F
    
    Lines:
    1. A, D, and intersection of BE and CF... 
    
    Let me just use a known Pappus configuration with integer coordinates.
    
    One standard Pappus configuration:
    Line 1 (y=0): (0,0), (1,0), (2,0)
    Line 2 (y=1): (0,1), (1,1), (2,1)
    
    The Pappus lines are:
    - Through (0,0) and (1,1): y=x
    - Through (1,0) and (0,1): x+y=1
    - Through (2,0) and (1,1): y=-x+2
    - Through (0,0) and (2,1): y=x/2
    - Through (1,0) and (0,1): x+y=1 (same as above)
    
    This is getting complicated. Let me use a different approach.
    
    I'll use the Hesse configuration or just do a smarter search.
    """
    pass


def _try_known_configs():
    """Try some known configurations that might have few ordinary lines."""
    configs = []
    
    # 3x4 grid
    grid = [[x, y] for x in range(3) for y in range(4)]
    configs.append(grid)
    
    # 4x3 grid
    grid2 = [[x, y] for x in range(4) for y in range(3)]
    configs.append(grid2)
    
    # 3 points on a line, 3 on another, etc.
    # Let's try a configuration with multiple collinear groups
    
    # 6 points on one line, 6 on another
    line1 = [[i, 0] for i in range(6)]
    line2 = [[i, 1] for i in range(6)]
    configs.append(line1 + line2)
    
    # 4 points on each of 3 parallel lines
    l1 = [[i, 0] for i in range(4)]
    l2 = [[i, 1] for i in range(4)]
    l3 = [[i, 2] for i in range(4)]
    configs.append(l1 + l2 + l3)
    
    # 3 points on each of 4 parallel lines
    l1 = [[i, 0] for i in range(3)]
    l2 = [[i, 1] for i in range(3)]
    l3 = [[i, 2] for i in range(3)]
    l4 = [[i, 3] for i in range(3)]
    configs.append(l1 + l2 + l3 + l4)
    
    return configs


def _local_search(points, deadline):
    """Perform local search to minimize ordinary lines."""
    if not _is_valid(points):
        return points
    
    current = [list(p) for p in points]
    best = [list(p) for p in points]
    best_score = _count_ordinary(best)
    current_score = best_score
    
    # Simulated annealing
    temp = 5.0
    cooling = 0.995
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_x, old_y = current[idx]
        
        # Random move
        if random.random() < 0.3:
            # Large jump
            new_x = random.randint(-50, 50)
            new_y = random.randint(-50, 50)
        else:
            # Small move
            new_x = old_x + random.randint(-5, 5)
            new_y = old_y + random.randint(-5, 5)
        
        new_point = [new_x, new_y]
        if any(new_point == p for p in current):
            continue
        
        current[idx] = new_point
        new_score = _count_ordinary(current)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in current]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, -delta / temp):
                current_score = new_score
            else:
                current[idx] = [old_x, old_y]
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try known configurations first
    best_pts = None
    best_score = float('inf')
    
    for config in _try_known_configs():
        if _is_valid(config):
            score = _count_ordinary(config)
            if score < best_score:
                best_score = score
                best_pts = [list(p) for p in config]
    
    # If no valid config found, use a default
    if best_pts is None:
        best_pts = [[i, 0] for i in range(12)]
        best_score = _count_ordinary(best_pts)
    
    # Local search from best known config
    result = _local_search(best_pts, deadline)
    
    # Verify
    if _is_valid(result):
        return result
    else:
        return best_pts
