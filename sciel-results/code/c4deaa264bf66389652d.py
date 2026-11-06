"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            g2 = gcd(abs(dx), abs(dy))
            g2 = gcd(g2, abs(c))
            if g2 > 0:
                dx_n = dx // g2
                dy_n = dy // g2
                c_n = c // g2
            else:
                dx_n, dy_n, c_n = dx, dy, c
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n, dy_n, c_n = -dx_n, -dy_n, -c_n
            key = (dx_n, dy_n, c_n)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    count = 0
    for pts_on_line in line_points.values():
        if len(pts_on_line) == 2:
            count += 1
    return count

def _local_search(points, max_time=10.0):
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _count_ordinary_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    n = len(current)
    max_coord = 10**9
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            orig_x, orig_y = current[i]
            # Try moving to a random position in a small grid
            for _ in range(50):
                new_x = random.randint(-10, 10)
                new_y = random.randint(-10, 10)
                if new_x == orig_x and new_y == orig_y:
                    continue
                current[i] = [new_x, new_y]
                if len(set(tuple(p) for p in current)) < n:
                    current[i] = [orig_x, orig_y]
                    continue
                score = _count_ordinary_lines(current)
                if score < current_score:
                    current_score = score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best = [list(p) for p in current]
                else:
                    current[i] = [orig_x, orig_y]
        if not improved:
            # Restart
            current = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(n)]
            if len(set(tuple(p) for p in current)) < n:
                continue
            current_score = _count_ordinary_lines(current)
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
    return best, best_score

def construct():
    # Start with a known good configuration or random
    # The 4x3 grid has ~10-12 ordinary lines.
    # A better start is a random set.
    best_points = None
    best_score = 100
    
    deadline = 140.0
    start = time.time()
    
    # Try a few random restarts
    for attempt in range(50):
        if time.time() - start > deadline:
            break
        random_pts = [[random.randint(-5, 5), random.randint(-5, 5)] for _ in range(12)]
        if len(set(tuple(p) for p in random_pts)) < 12:
            continue
        pts, score = _local_search(random_pts, max_time=2.0)
        if score < best_score:
            best_score = score
            best_points = pts
            if best_score == 6:
                break
                
    # If we didn't find 6, try the 4x3 grid and optimize
    if best_score > 6:
        grid = [[x, y] for x in range(4) for y in range(3)]
        pts, score = _local_search(grid, max_time=min(10.0, deadline - (time.time() - start)))
        if score < best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1], [2,1], [3,1], [0,2], [1,2], [2,2], [3,2]]
        
    return best_points
# EVOLVE-BLOCK-END
