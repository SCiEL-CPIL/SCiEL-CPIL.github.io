import time
import random
from itertools import combinations
from math import gcd
import sys

def _line_through_two(p1, p2):
    """Return normalized line (dx, dy, c) for line through p1, p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    # Normalize sign
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * p1[0] - dy * p1[1]
    return (dx, dy, c)

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_to_pts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_through_two(points[i], points[j])
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    
    count = 0
    for key, pts in line_to_pts.items():
        if len(pts) == 4:
            count += 1
    return count

def _no_five_collinear(points):
    """Check that no 5 points are collinear."""
    n = len(points)
    line_to_pts = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_through_two(points[i], points[j])
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    for key, pts in line_to_pts.items():
        if len(pts) >= 5:
            return False
    return True

def _score(points):
    """Return score: number of 4-point lines if valid, else -1."""
    if len(points) != 16:
        return -1
    if len(set(tuple(p) for p in points)) != 16:
        return -1
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return -1
    if not _no_five_collinear(points):
        return -1
    return _count_lines_exactly_4(points)

def _random_point(rng, limit):
    return [rng.randint(-limit, limit), rng.randint(-limit, limit)]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    rng = random.Random(42)
    
    # Initial candidates
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _score(best_points)
    
    # Simulated Annealing
    # We will run multiple restarts or a long SA chain
    
    current_points = [p[:] for p in best_points]
    current_score = best_score
    
    # Search space limit
    limit = 20
    
    # Parameters
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Pre-compute some valid moves? No, dynamic.
    
    while time.time() < deadline and T > min_T:
        # Pick a random point to move
        idx = rng.randint(0, 15)
        orig_point = current_points[idx]
        
        # Generate a new point
        # Bias towards small integers
        new_point = _random_point(rng, limit)
        
        # Check if new_point is already in the set
        if new_point == orig_point:
            continue
        if any(new_point == p for p in current_points):
            continue
            
        # Try the move
        old_point = current_points[idx]
        current_points[idx] = new_point
        
        new_score = _score(current_points)
        
        if new_score < 0:
            # Invalid, revert
            current_points[idx] = old_point
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < pow(2.71828, delta / T):
            # Accept
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]
        else:
            # Reject
            current_points[idx] = old_point
            
        T *= cooling
        
        # Occasionally reset T to escape local optima
        if rng.random() < 0.01:
            T = max(T, 0.5)
            
    # If we still have time, try random restarts
    while time.time() < deadline - 5.0:
        # Random start
        points = []
        attempts = 0
        while len(points) < 16 and attempts < 1000:
            p = _random_point(rng, 10)
            if p not in points:
                points.append(p)
            attempts += 1
        
        if len(points) < 16:
            continue
            
        # Quick local search
        p_score = _score(points)
        if p_score < 0:
            continue
            
        # Short SA
        T_local = 0.5
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            idx = rng.randint(0, 15)
            old_p = points[idx]
            new_p = _random_point(rng, 10)
            if new_p in points:
                continue
            points[idx] = new_p
            ns = _score(points)
            if ns < 0:
                points[idx] = old_p
                continue
            d = ns - p_score
            if d > 0 or rng.random() < pow(2.71828, d / T_local):
                p_score = ns
            else:
                points[idx] = old_p
            T_local *= 0.99
            
        if p_score > best_score:
            best_score = p_score
            best_points = [p[:] for p in points]
            
    return best_points
