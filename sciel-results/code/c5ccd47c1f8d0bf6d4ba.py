import numpy as np
import time
import itertools

def erdos_szekeres_problem7() -> np.ndarray:
    n = 33
    k = 7
    start_time = time.time()
    deadline = start_time + 580

    def is_convex_7(p):
        # p is (7, 2)
        # Check if all turns are same direction (e.g., left)
        # and non-zero area
        cross = 0
        for i in range(7):
            x1, y1 = p[i]
            x2, y2 = p[(i+1)%7]
            x3, y3 = p[(i+2)%7]
            c = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
            if c < 0:
                return False
            if abs(c) < 1e-9:
                return False
        return True

    def count_convex_7(points):
        count = 0
        # Use combinations
        for combo in itertools.combinations(range(n), 7):
            subset = points[list(combo)]
            if is_convex_7(subset):
                count += 1
        return count

    # Initialization: Cup-Cap
    points = np.zeros((n, 2))
    for i in range(16):
        x = i / 15.0 * 0.5
        points[i, 0] = x
        points[i, 1] = x * x
        
    for i in range(16):
        x = 0.5 + i / 15.0 * 0.5
        points[16+i, 0] = x
        points[16+i, 1] = 1 - (x - 1) ** 2
        
    points[32, 0] = 0.5
    points[32, 1] = 0.5

    # Local Search
    best_points = points.copy()
    best_count = count_convex_7(best_points)
    
    while time.time() < deadline:
        # Move a random point
        idx = np.random.randint(n)
        old_x, old_y = best_points[idx]
        
        # Random new position
        new_x = np.random.rand()
        new_y = np.random.rand()
        
        best_points[idx, 0] = new_x
        best_points[idx, 1] = new_y
        
        new_count = count_convex_7(best_points)
        
        if new_count < best_count:
            best_count = new_count
        else:
            best_points[idx, 0] = old_x
            best_points[idx, 1] = old_y

    return best_points
