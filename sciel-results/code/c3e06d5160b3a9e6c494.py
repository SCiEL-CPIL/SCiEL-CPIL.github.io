import numpy as np
import time
import random
from itertools import combinations

N = 64
TIME_LIMIT = 320  # seconds, safety margin before 330s kill

def _squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def _is_valid_set(points):
    """Check if the set of points has no three forming an isosceles triangle."""
    n = len(points)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d_ij = _squared_dist(points[i], points[j])
                d_ik = _squared_dist(points[i], points[k])
                d_jk = _squared_dist(points[j], points[k])
                if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                    return False
    return True

def _can_add_point(point, points):
    """Check if adding point to points keeps the set valid."""
    for i, p1 in enumerate(points):
        d1 = _squared_dist(point, p1)
        for j in range(i+1, len(points)):
            p2 = points[j]
            d2 = _squared_dist(point, p2)
            d12 = _squared_dist(p1, p2)
            if d1 == d2 or d1 == d12 or d2 == d12:
                return False
    return True

def _greedy_random(points_order, existing):
    """Greedily add points from points_order to existing set."""
    result = list(existing)
    for p in points_order:
        if _can_add_point(p, result):
            result.append(p)
    return result

def _local_search(points, time_limit):
    """Local search: try removing one point and adding others."""
    start_time = time.time()
    best = list(points)
    best_size = len(best)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try each point to remove
        for i in range(len(best)):
            if time.time() - start_time > time_limit:
                break
            candidate = best[:i] + best[i+1:]
            # Try to add other points
            for j in range(len(best)):
                if j == i:
                    continue
                p = best[j]
                # Already in candidate, skip
            # Try random other grid points
            for _ in range(20):
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p not in candidate and _can_add_point(p, candidate):
                    new_set = candidate + [p]
                    if len(new_set) > len(candidate):
                        best = new_set
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try different removals or accept
            break
    
    return best

def construct():
    start_time = time.time()
    
    # Generate all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_result = []
    
    # Multiple random restarts with greedy
    for restart in range(50):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Random order with some structure
        random.shuffle(all_points)
        
        # Greedy construction
        current = []
        for p in all_points:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            if _can_add_point(p, current):
                current.append(p)
        
        if len(current) > len(best_result):
            best_result = current
        
        # Local search on the best
        if len(current) >= len(best_result) and len(current) > 5:
            improved = _local_search(current, TIME_LIMIT - 60 - (time.time() - start_time))
            if len(improved) > len(best_result):
                best_result = improved
    
    return best_result
