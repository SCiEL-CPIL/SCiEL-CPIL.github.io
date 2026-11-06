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
    # Map from normalized line to set of point indices on it
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
            # Normalize sign
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line equation: dy*(x - x1) - dx*(y - y1) = 0
            # => dy*x - dx*y = dy*x1 - dx*y1
            c = dy * x1 - dx * y1
            # Normalize (dx, dy, c)
            g2 = gcd(abs(dx), abs(dy))
            g2 = gcd(g2, abs(c))
            if g2 > 0:
                dx_n = dx // g2
                dy_n = dy // g2
                c_n = c // g2
            else:
                dx_n, dy_n, c_n = dx, dy, c
            # Ensure canonical sign
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

def _local_search(points, max_time=100.0):
    """Local search to minimize ordinary lines."""
    start_time = time.time()
    current = [list(p) for p in points]
    current_score = _count_ordinary_lines(current)
    best = [list(p) for p in current]
    best_score = current_score
    n = len(current)
    max_coord = 10**9
    # Use a smaller search space for efficiency
    search_range = 50
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(n):
            # Try moving point i
            orig_x, orig_y = current[i]
            # Try random new positions
            for _ in range(200):
                new_x = random.randint(-search_range, search_range)
                new_y = random.randint(-search_range, search_range)
                if new_x == orig_x and new_y == orig_y:
                    continue
                # Check distinctness
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
            # Also try small perturbations of original
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if abs(new_x) > max_coord or abs(new_y) > max_coord:
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
            # Restart with random perturbation
            for i in range(n):
                current[i] = [random.randint(-search_range, search_range),
                              random.randint(-search_range, search_range)]
            if len(set(tuple(p) for p in current)) < n:
                continue
            current_score = _count_ordinary_lines(current)
            if current_score < best_score:
                best_score = current_score
                best = [list(p) for p in current]
    
    return best, best_score

def construct():
    # Start with 3x4 grid
    initial = [[x, y] for x in range(4) for y in range(3)]
    best_points = initial
    best_score = _count_ordinary_lines(initial)
    
    # Try local search
    deadline = 100.0
    start = time.time()
    
    # Multiple restarts
    for attempt in range(20):
        if time.time() - start > deadline:
            break
        # Random initial configuration
        random_pts = [[random.randint(-30, 30), random.randint(-30, 30)] for _ in range(12)]
        if len(set(tuple(p) for p in random_pts)) < 12:
            continue
        pts, score = _local_search(random_pts, max_time=3.0)
        if score < best_score:
            best_score = score
            best_points = pts
    
    # Also try from the grid
    if time.time() - start < deadline:
        pts, score = _local_search(initial, max_time=min(10.0, deadline - (time.time() - start)))
        if score < best_score:
            best_score = score
            best_points = pts
    
    # Check if we can do better with a structured approach
    # Try 2x6 grid
    grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
    s = _count_ordinary_lines(grid_2x6)
    if s < best_score:
        best_score = s
        best_points = grid_2x6
    
    # Try 1x12 line (all collinear - invalid, skip)
    # Try other small grids
    for h in range(2, 5):
        for w in range(2, 7):
            if h * w == 12:
                g = [[x, y] for x in range(w) for y in range(h)]
                s = _count_ordinary_lines(g)
                if s < best_score:
                    best_score = s
                    best_points = g
    
    return best_points
# EVOLVE-BLOCK-END
