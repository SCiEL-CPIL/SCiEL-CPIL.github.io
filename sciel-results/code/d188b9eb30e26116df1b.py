"""Number-theoretic construction: lattice points in a parallelogram to maximize area multiplicity."""
import itertools
from collections import Counter

def construct():
    # Use a 2x15 grid structure to maximize area multiplicity.
    # This corresponds to a generalized arithmetic progression with basis vectors u=(1,0) and v=(0,1).
    # The thin aspect ratio concentrates area values, specifically the smallest non-zero area,
    # leading to a higher maximum multiplicity than more square-like grids.
    
    # Points: x in {0, 1}, y in {0, ..., 14}
    # Shift y to center around 0 for symmetry within [-30, 30]
    points = []
    for i in range(2):
        for j in range(15):
            x = i
            y = j - 7  # Range -7 to 7
            points.append((x, y))
    
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
