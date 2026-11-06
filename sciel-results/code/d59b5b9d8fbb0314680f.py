"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    # Map each line (defined by two points) to the set of points on it
    # Use a normalized line representation: (A, B, C) for Ax + By + C = 0
    # with gcd normalization and sign convention
    line_points = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2): (y2-y1)x - (x2-x1)y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            # Normalize
            if A == 0 and B == 0:
                continue
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            # Divide by gcd
            g = _gcd3(A, B, C)
            if g > 1:
                A //= g
                B //= g
                C //= g
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _gcd3(a, b, c):
    from math import gcd
    g = gcd(abs(a), abs(b))
    g = gcd(g, abs(c))
    return g

def _generate_grid_config(rows, cols, x_start=0, y_start=0, x_step=1, y_step=1):
    pts = []
    for i in range(rows):
        for j in range(cols):
            pts.append([x_start + j * x_step, y_start + i * y_step])
    return pts

def _perturb_points(points, magnitude=5):
    pts = [p[:] for p in points]
    for p in pts:
        p[0] += random.randint(-magnitude, magnitude)
        p[1] += random.randint(-magnitude, magnitude)
    return pts

def _local_search(points, max_iters=100, time_limit=None):
    """Local search: try moving individual points to reduce ordinary line count."""
    current = [p[:] for p in points]
    current_score = _count_ordinary_lines(current)
    n = len(current)
    
    improved = True
    iters = 0
    start_time = time.time()
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        # Try moving each point
        for i in range(n):
            orig = current[i][:]
            best_new = None
            best_score = current_score
            # Smaller neighborhood for speed
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    pos = (orig[0] + dx, orig[1] + dy)
                    if pos == (orig[0], orig[1]):
                        continue
                    # Check bounds
                    if abs(pos[0]) > 10**9 or abs(pos[1]) > 10**9:
                        continue
                    # Check distinctness
                    new_pts = [p[:] for p in current]
                    new_pts[i] = [pos[0], pos[1]]
                    # Check if any other point is at this position
                    dup = False
                    for j in range(n):
                        if j != i and new_pts[j] == new_pts[i]:
                            dup = True
                            break
                    if dup:
                        continue
                    score = _count_ordinary_lines(new_pts)
                    if score < best_score:
                        best_score = score
                        best_new = [pos[0], pos[1]]
            if best_new is not None:
                current[i] = best_new
                current_score = best_score
                improved = True
    
    return current, current_score

def construct():
    deadline = time.time() + 140  # Leave 10s margin
    best_points = None
    best_score = float('inf')
    
    # Initial configurations
    configs = []
    # 3x4 grid
    configs.append(_generate_grid_config(3, 4))
    # 4x3 grid
    configs.append(_generate_grid_config(4, 3))
    # 2x6 grid
    configs.append(_generate_grid_config(2, 6))
    # 6x2 grid
    configs.append(_generate_grid_config(6, 2))
    # Perturbed grids
    random.seed(42)
    for _ in range(20):
        base = _generate_grid_config(random.choice([2,3,4]), random.choice([3,4,6]))
        configs.append(_perturb_points(base, random.randint(1, 20)))
    
    for config in configs:
        if time.time() > deadline:
            break
        # Ensure distinct points
        seen = set()
        valid = True
        for p in config:
            tp = tuple(p)
            if tp in seen or abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                valid = False
                break
            seen.add(tp)
        if not valid or len(seen) != 12:
            continue
        # Check not all collinear
        if _all_collinear(config):
            continue
        
        pts, score = _local_search(config, max_iters=500, time_limit=5.0)
        if score < best_score:
            best_score = score
            best_points = pts
        if best_score <= 6:
            break
    
    if best_points is None:
        # Fallback: return the 3x4 grid
        return [[x, y] for x in range(4) for y in range(3)]
    
    return best_points

def _all_collinear(points):
    if len(points) < 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    if cross != 0:
        return False
    for i in range(3, len(points)):
        xi, yi = points[i]
        if (x1 - x0) * (yi - y0) - (y1 - y0) * (xi - x0) != 0:
            return False
    return True
# EVOLVE-BLOCK-END
