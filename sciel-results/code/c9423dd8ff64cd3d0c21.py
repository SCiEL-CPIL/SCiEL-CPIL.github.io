import itertools
import random
from math import gcd

def _count_ordinary_lines(points):
    """Count the number of lines that contain exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(A, gcd(B, C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                A, B, C = 0, 0, 0
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            lines.add((A, B, C))
    
    ordinary = 0
    for line in lines:
        A, B, C = line
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    return ordinary

def construct():
    # Start with a structured configuration
    # Try a 4x3 grid
    base_points = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1], [3, 1],
        [0, 2], [1, 2], [2, 2], [3, 2]
    ]
    
    best_points = [p[:] for p in base_points]
    best_count = _count_ordinary_lines(best_points)
    
    # Local search with random restarts
    import random
    random.seed(42)
    
    for _ in range(100):
        # Start from best or random perturbation
        current_points = [p[:] for p in best_points]
        if _ > 0 and random.random() < 0.5:
            # Random restart
            current_points = []
            for _ in range(12):
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                current_points.append([x, y])
            # Ensure distinct
            while len(set(tuple(p) for p in current_points)) < 12:
                idx = random.randint(0, 11)
                current_points[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        
        current_count = _count_ordinary_lines(current_points)
        
        improved = True
        while improved:
            improved = False
            # Try moving each point
            for idx in range(12):
                x0, y0 = current_points[idx]
                # Try neighborhood
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x0 + dx, y0 + dy
                        new_points = [p[:] for p in current_points]
                        new_points[idx] = [nx, ny]
                        
                        if len(set(tuple(p) for p in new_points)) != 12:
                            continue
                        
                        # Check not all collinear
                        collinear = True
                        for i in range(1, 12):
                            if not (new_points[i][0] - new_points[0][0]) * (new_points[1][1] - new_points[0][1]) == \
                               (new_points[i][1] - new_points[0][1]) * (new_points[1][0] - new_points[0][0]):
                                collinear = False
                                break
                        if collinear:
                            continue
                        
                        cnt = _count_ordinary_lines(new_points)
                        if cnt < current_count:
                            current_count = cnt
                            current_points = new_points
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if current_count < best_count:
                best_count = current_count
                best_points = [p[:] for p in current_points]
                if best_count == 6:
                    break
    
    return best_points
