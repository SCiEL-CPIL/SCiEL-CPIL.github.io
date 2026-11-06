import time
import random
from itertools import combinations
from math import gcd

def count_exact_four_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    count = 0
    for line, pts in line_to_points.items():
        if len(pts) == 4:
            count += 1
    
    return count

def has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    if n < 5:
        return False
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    for line, pts in line_to_points.items():
        if len(pts) >= 5:
            return True
    
    return False

def is_valid(points):
    """Check all constraints: distinct, bounds, no 5 collinear."""
    if len(points) != 16:
        return False
    
    # Check distinct
    seen = set()
    for p in points:
        t = tuple(p)
        if t in seen:
            return False
        seen.add(t)
    
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    
    # Check no 5 collinear
    if has_five_collinear(points):
        return False
    
    return True

def evaluate(points):
    """
    Returns (is_valid, score).
    Combines validity check and scoring.
    """
    n = len(points)
    if n != 16:
        return False, 0
        
    seen = set()
    for p in points:
        x, y = p
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
        t = (x, y)
        if t in seen:
            return False, 0
        seen.add(t)
        
    line_to_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            a = y2 - y1
            b = -(x2 - x1)
            c = (x2 - x1) * y1 - (y2 - y1) * x2
            
            if a == 0 and b == 0:
                continue
            g = gcd(abs(a), abs(b))
            if g != 0:
                a //= g
                b //= g
                c //= g
            
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            
            key = (a, b, c)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
            
    score = 0
    for line, pts in line_to_points.items():
        if len(pts) == 4:
            score += 1
        elif len(pts) >= 5:
            return False, 0
            
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    random.seed(42)
    
    # Initial configuration: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    valid, score = evaluate(points)
    
    best_points = [p[:] for p in points]
    best_score = score
    
    # Local Search with expanding neighborhoods
    # We try to move each point to a better position.
    # Neighborhoods are defined by coordinate ranges.
    
    # Define search ranges. Start tight, expand if stuck.
    ranges = [
        (-2, 5),    # Small local moves
        (-5, 10),   # Medium local moves
        (-10, 20),  # Larger moves
        (-20, 50),  # Large moves
        (-50, 100), # Very large moves
        (-100, 200) # Huge moves
    ]
    
    for r_low, r_high in ranges:
        improved = True
        while improved:
            improved = False
            for i in range(16):
                old_x, old_y = points[i]
                best_local_score = score
                best_local_pos = [old_x, old_y]
                
                # Generate candidates
                # If range is small, enumerate all. If large, sample.
                range_size = r_high - r_low
                if range_size <= 20:
                    candidates = [[nx, ny] for nx in range(r_low, r_high + 1) for ny in range(r_low, r_high + 1)]
                else:
                    # Sample 200 random points in the box
                    candidates = []
                    for _ in range(200):
                        nx = random.randint(r_low, r_high)
                        ny = random.randint(r_low, r_high)
                        candidates.append([nx, ny])
                
                for nx, ny in candidates:
                    if nx == old_x and ny == old_y:
                        continue
                    # Quick check: is it occupied?
                    if [nx, ny] in points:
                        continue
                        
                    new_points = [p[:] for p in points]
                    new_points[i] = [nx, ny]
                    
                    v, s = evaluate(new_points)
                    if v and s > best_local_score:
                        best_local_score = s
                        best_local_pos = [nx, ny]
                        
                if best_local_pos != [old_x, old_y]:
                    points[i] = best_local_pos
                    score = best_local_score
                    improved = True
                    if score > best_score:
                        best_score = score
                        best_points = [p[:] for p in points]
                        
            if time.time() > deadline - 5:
                break
                
        if time.time() > deadline - 5:
            break
            
    return best_points
