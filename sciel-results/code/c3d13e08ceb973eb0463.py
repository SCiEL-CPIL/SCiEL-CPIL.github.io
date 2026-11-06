import time
import math

def construct():
    # Direct construction of optimal hexagonal spiral for 32 points
    # Harborth's bound: 76 unit distances, attained by hex spiral
    
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a+b)) // 2
    
    # Generate all points with hex_dist <= 2 (complete inner hexagon: 19 points)
    inner = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            if hex_dist(a, b) <= 2:
                inner.append((a, b))
    
    # Generate all points with hex_dist == 3 (ring 3: 18 points)
    outer = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if hex_dist(a, b) == 3:
                outer.append((a, b))
    
    # Need 32 - 19 = 13 points from outer ring
    # Sort by angle to get spiral order
    def angle(p):
        a, b = p
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    outer_sorted = sorted(outer, key=angle)
    
    # Try all combinations of 13 from 18 outer points to maximize edges
    # C(18,13) = 8568, very manageable
    from itertools import combinations
    
    neighbors = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    def count_edges(points):
        point_set = set(points)
        count = 0
        for a, b in points:
            for da, db in neighbors:
                if (a+da, b+db) in point_set:
                    count += 1
        return count // 2
    
    best_count = 0
    best_outer_subset = None
    
    inner_set = set(inner)
    
    for subset in combinations(outer_sorted, 13):
        points = inner + list(subset)
        # Check for duplicates (shouldn't happen since inner and outer are disjoint)
        if len(set(points)) != 32:
            continue
        c = count_edges(points)
        if c > best_count:
            best_count = c
            best_outer_subset = subset
    
    # If we found 76, we're done
    if best_count >= 76:
        points = inner + list(best_outer_subset)
        return [[a, b] for a, b in points]
    
    # Fallback: use the spiral-ordered 13 points
    points = inner + outer_sorted[:13]
    return [[a, b] for a, b in points]
