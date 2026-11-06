import math
import time

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance edges (target: 76).
    """
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate points in hexagonal spiral order (by distance from origin, then angle)
    points = []
    seen = set()
    
    # Generate candidates in a box large enough to contain the optimal configuration
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            d2 = a*a + a*b + b*b
            angle = math.atan2(b, a)
            candidates.append((d2, angle, a, b))
    
    # Sort by distance squared, then by angle for stable ordering
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Select first 32 distinct points in spiral order
    for d2, angle, a, b in candidates:
        if (a, b) not in seen:
            seen.add((a, b))
            points.append([a, b])
            if len(points) == 32:
                break
    
    # Fallback if something went wrong
    if len(points) != 32:
        points = [
            [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1],
            [2, 0], [1, 1], [2, -1], [-2, 0], [-1, -1], [-2, 1], [0, 2],
            [-1, 2], [0, -2], [1, -2], [2, -2], [-2, 2], [3, 0], [2, 1],
            [3, -1], [1, 2], [3, -2], [-3, 0], [-2, -1], [-3, 1], [-1, -2],
            [-3, 2], [0, 3], [-1, 3], [-2, 3]
        ]
    
    def count_edges(pts):
        count = 0
        n = len(pts)
        for i in range(n):
            a1, b1 = pts[i]
            for j in range(i+1, n):
                a2, b2 = pts[j]
                da = a1 - a2
                db = b1 - b2
                if da*da + da*db + db*db == 1:
                    count += 1
        return count
    
    current_edges = count_edges(points)
    
    # Local search: try moving each point to an adjacent empty lattice position
    # if it increases the edge count
    if current_edges < 76:
        point_set = set(tuple(p) for p in points)
        
        for iteration in range(10):  # Multiple passes
            if time.time() > deadline:
                break
            improved = False
            for i in range(32):
                if time.time() > deadline:
                    break
                ai, bi = points[i]
                # Try the 6 nearest neighbors on triangular lattice
                neighbors = [(ai+1, bi), (ai-1, bi), (ai, bi+1), 
                            (ai, bi-1), (ai+1, bi-1), (ai-1, bi+1)]
                for na, nb in neighbors:
                    if (na, nb) in point_set:
                        continue
                    # Count edges lost by removing point i
                    edges_lost = 0
                    for j in range(32):
                        if j == i:
                            continue
                        aj, bj = points[j]
                        da = ai - aj
                        db = bi - bj
                        if da*da + da*db + db*db == 1:
                            edges_lost += 1
                    # Count edges gained by placing at new position
                    edges_gained = 0
                    for j in range(32):
                        if j == i:
                            continue
                        aj, bj = points[j]
                        da = na - aj
                        db = nb - bj
                        if da*da + da*db + db*db == 1:
                            edges_gained += 1
                    if edges_gained > edges_lost:
                        points[i] = [na, nb]
                        point_set.discard((ai, bi))
                        point_set.add((na, nb))
                        current_edges += (edges_gained - edges_lost)
                        improved = True
                        break
            if not improved:
                break
    
    return points
