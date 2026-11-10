import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    # Construction with known score 14
    # 6 disjoint lines of 3 points
    # Points are chosen to be well-separated to avoid accidental 4-collinearities
    pts = [
        [10, 0], [11, 0], [12, 0],
        [-12, 2], [-11, 2], [-10, 2],
        [10, 4], [11, 4], [12, 4],
        [-12, 6], [-11, 6], [-10, 6],
        [10, 8], [11, 8], [12, 8],
        [-12, 10], [-11, 10], [-10, 10]
    ]
    # 2 isolated points
    pts.append([50, 50])
    pts.append([-50, -50])
    
    def is_valid(points):
        if len(points) != 20:
            return False
        if len(set(map(tuple, points))) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        
        # Check no 4 collinear
        for comb in itertools.combinations(points, 4):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = comb
            d1 = (x2 - x1) * (y3 - y1)
            d2 = (x3 - x1) * (y2 - y1)
            if d1 != d2:
                continue
            d3 = (x2 - x1) * (y4 - y1)
            d4 = (x4 - x1) * (y2 - y1)
            if d3 != d4:
                continue
            return False
        return True

    if not is_valid(pts):
        # Fallback to a general position set (score 20) if construction fails
        # This is unlikely to happen
        return [[x, x * x] for x in range(-10, 10)]
        
    return pts
