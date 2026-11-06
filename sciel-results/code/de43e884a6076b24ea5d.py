import itertools
from collections import Counter

def construct():
    # Use 2 parallel lines to maximize the frequency of the smallest area.
    # Points on line 1: (x, 0) for x in [0, 14]
    # Points on line 2: (x, 1) for x in [0, 14]
    # This creates 30 distinct points.
    
    n1 = 15
    n2 = 15
    
    points = []
    for i in range(n1):
        points.append((i, 0))
    for i in range(n2):
        points.append((i, 1))
        
    # Verify 30 distinct points in [-30, 30]^2
    assert len(set(points)) == 30
    assert all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in points)
    
    # Compute absolute determinants for every triple
    areas = []
    for i in range(30):
        for j in range(i+1, 30):
            for k in range(j+1, 30):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) - x2*(y1-y3) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    
    # Find the largest multiplicity
    counter = Counter(areas)
    max_mult = max(counter.values()) if counter else 0
    
    return points
