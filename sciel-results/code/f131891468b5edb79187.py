import time
import random
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction vector
            if dx == 0 and dy == 0:
                continue
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            elif dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = _gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
            # Compute a canonical line identifier: (slope, intercept) in a normalized form
            # Line: dy*x - dx*y = c
            c = dy * x1 - dx * y1
            lines.add((dx, dy, c))
    
    # For each line, count how many points lie on it
    ordinary = 0
    for (dx, dy, c) in lines:
        cnt = 0
        for x, y in points:
            if dy * x - dx * y == c:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    return ordinary

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _evaluate(points):
    """Return the number of ordinary lines for the given points."""
    return _count_ordinary_lines(points)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Initial known good configuration: 3x4 grid gives 8 ordinary lines
    # Let's verify and then try to improve
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _evaluate(best_points)
    
    # Try some known configurations
    # Configuration 1: 3x4 grid (current)
    # Configuration 2: Try a more structured approach
    
    # Let's try a 2x6 arrangement
    configs = [
        [[x, y] for x in range(4) for y in range(3)],  # 3x4 grid
        [[x, y] for x in range(6) for y in range(2)],  # 2x6 grid
        # Try a hexagonal-like pattern
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2],[4,2]],
        # Try a 4x3 with some modifications
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],
        # Try points on a conic or other structure
    ]
    
    for config in configs:
        if len(config) != 12:
            continue
        score = _evaluate(config)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in config]
    
    # Local search: try to improve by moving individual points
    def _local_search(points, deadline):
        points = [p[:] for p in points]
        current_score = _evaluate(points)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(12):
                orig_x, orig_y = points[i]
                # Try small perturbations
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                            continue
                        # Check if point already exists
                        if [new_x, new_y] in points:
                            continue
                        old_pt = points[i]
                        points[i] = [new_x, new_y]
                        new_score = _evaluate(points)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            points[i] = old_pt
        
        return points, current_score
    
    best_points, best_score = _local_search(best_points, deadline)
    
    # Try multiple random restarts if time permits
    if time.time() < deadline - 30.0:
        random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Generate a random configuration
            pts = []
            while len(pts) < 12:
                x = random.randint(-5, 5)
                y = random.randint(-5, 5)
                if [x, y] not in pts:
                    pts.append([x, y])
            
            # Ensure not all collinear
            if _all_collinear(pts):
                continue
            
            score = _evaluate(pts)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in pts]
            
            # Local search
            if time.time() < deadline - 10.0:
                pts2, score2 = _local_search(pts, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_points = [p[:] for p in pts2]
    
    # Final check
    assert len(best_points) == 12
    assert all(len(p) == 2 for p in best_points)
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in best_points)
    assert len(set(tuple(p) for p in best_points)) == 12
    assert not _all_collinear(best_points)
    
    return best_points

def _all_collinear(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    return (y1 - y0) * (x2 - x0) == (y2 - y0) * (x1 - x0)
