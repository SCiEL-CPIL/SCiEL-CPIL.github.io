"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from math import gcd
import random

def count_two_point_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_to_points = {}
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            if A < 0 or (A == 0 and B < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            if key not in line_to_points:
                line_to_points[key] = set()
            line_to_points[key].add(i)
            line_to_points[key].add(j)
    
    two_point_lines = sum(1 for pts in line_to_points.values() if len(pts) == 2)
    return two_point_lines

def all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    for i in range(2, len(points)):
        x3, y3 = points[i]
        if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
            return False
    return True

def construct():
    """
    Construct 12 points minimizing the number of 2-point lines.
    Uses a backtracking search that grows the point set by adding
    points on existing lines to maximize 3+ point lines.
    """
    
    N = 12
    grid_size = 10
    grid_points = set((x, y) for x in range(grid_size) for y in range(grid_size))
    
    best_score = float('inf')
    best_config = None
    
    def greedy_build(initial_points):
        points = [tuple(p) for p in initial_points]
        if len(set(points)) < 3:
            return None, float('inf')
        
        used = set(points)
        
        for _ in range(N - len(points)):
            best_candidate = None
            best_candidate_score = None
            
            for p in grid_points:
                if p in used:
                    continue
                
                test_points = list(used) + [p]
                score = count_two_point_lines(test_points)
                
                if best_candidate_score is None or score < best_candidate_score:
                    best_candidate_score = score
                    best_candidate = p
            
            if best_candidate is None:
                return None, float('inf')
            
            points.append(best_candidate)
            used.add(best_candidate)
        
        return [list(p) for p in points], count_two_point_lines(points)

    random.seed(42)
    for _ in range(100):
        initial = random.sample(list(grid_points), 3)
        x1, y1 = initial[0]
        x2, y2 = initial[1]
        x3, y3 = initial[2]
        if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
            continue
            
        config, score = greedy_build(initial)
        if config and score < best_score:
            best_score = score
            best_config = config
            
    if best_config is None:
        best_config = [[x, y] for x in range(4) for y in range(3)]
        
    return best_config

# EVOLVE-BLOCK-END
