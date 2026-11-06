"""Number-theoretic construction: lattice points in a parallelogram to maximize area multiplicity."""
import itertools
from collections import Counter

def construct():
    # We want 30 points. A 5x6 rectangular grid is a strong candidate.
    # Points: (x, y) for x in [0, 4], y in [0, 5]
    # Shift to center within [-30, 30]^2
    points = [(x - 2, y - 2) for x in range(5) for y in range(6)]
    
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
