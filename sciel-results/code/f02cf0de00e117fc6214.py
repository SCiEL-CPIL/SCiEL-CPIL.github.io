"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly two of the points."""
    n = len(points)
    line_to_pts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            # Normalize the direction vector
            if dx == 0:
                # vertical line
                key = (0, 0, xi)  # (dx, dy, x)
            elif dy == 0:
                # horizontal line
                key = (0, 0, yi)
            else:
                # Normalize (dx, dy) by gcd
                import math
                g = math.gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                # Normalize sign so that dx_n > 0 or (dx_n == 0 and dy_n > 0)
                if dx_n < 0:
                    dx_n = -dx_n
                    dy_n = -dy_n
                elif dx_n == 0 and dy_n < 0:
                    dy_n = -dy_n
                # For non-vertical, non-horizontal lines, use (dx_n, dy_n, c)
                # where c = y - (dy_n/dx_n) * x  =>  dy_n * x - dx_n * y = c
                # c = dy_n * xi - dx_n * yi
                c = dy_n * xi - dx_n * yi
                key = (dx_n, dy_n, c)
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    
    ordinary = 0
    for key, pts in line_to_pts.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _brute_force_search(max_coord=20, time_limit=140):
    """Search for 12-point configurations with few ordinary lines."""
    import random
    import math
    
    start_time = time.time()
    best_score = float('inf')
    best_points = None
    
    # First, try some structured configurations
    # 1. Hesse-like configuration (3x4 grid variant)
    structured_configs = [
        # 3x4 grid
        [[x, y] for x in range(4) for y in range(3)],
        # 4x3 grid
        [[x, y] for x in range(3) for y in range(4)],
        # Points on two parallel lines
        [[x, 0] for x in range(6)] + [[x, 1] for x in range(6)],
        # Points on three parallel lines with 4 each
        [[x, 0] for x in range(4)] + [[x, 1] for x in range(4)] + [[x, 2] for x in range(4)],
        # Points on four parallel lines with 3 each
        [[x, 0] for x in range(3)] + [[x, 1] for x in range(3)] + [[x, 2] for x in range(3)] + [[x, 3] for x in range(3)],
        # 2x6 grid
        [[x, y] for x in range(6) for y in range(2)],
        # 6x2 grid
        [[x, y] for x in range(2) for y in range(6)],
        # Points on a circle (integer approximations)
        # Using Pythagorean triples
        [[3, 4], [4, 3], [-3, 4], [-4, 3], [3, -4], [4, -3], [-3, -4], [-4, -3],
         [5, 12], [12, 5], [-5, 12], [-12, 5]],
        # Another set
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
         [2, 0], [-2, 0], [0, 2], [0, -2]],
    ]
    
    for config in structured_configs:
        if len(config) != 12:
            continue
        # Check all distinct
        if len(set(map(tuple, config))) != 12:
            continue
        # Check bounds
        if any(abs(x) > max_coord or abs(y) > max_coord for x, y in config):
            continue
        # Check not all collinear
        if all((config[i][0] - config[0][0]) * (config[i][1] - config[0][1]) == 
               (config[i+1][0] - config[0][0]) * (config[i+1][1] - config[0][1]) 
               for i in range(11)):
            continue
        score = _count_ordinary_lines(config)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in config]
    
    # If we found 6, we're done (theoretical minimum)
    if best_score <= 6:
        return best_points
    
    # Random search
    random.seed(42)
    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        # Generate random points
        pts = []
        while len(pts) < 12:
            x = random.randint(-max_coord, max_coord)
            y = random.randint(-max_coord, max_coord)
            if (x, y) not in [(p[0], p[1]) for p in pts]:
                pts.append([x, y])
        
        # Check not all collinear
        collinear = True
        for i in range(1, 12):
            if (pts[i][0] - pts[0][0]) * (pts[i][1] - pts[0][1]) != \
               (pts[i-1][0] - pts[0][0]) * (pts[i-1][1] - pts[0][1]):
                collinear = False
                break
        if collinear:
            continue
        
        score = _count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            if best_score <= 6:
                break
        
        # Early exit if time is running out
        if time.time() - start_time > time_limit - 5:
            break
    
    return best_points

def construct():
    """Return exactly 12 distinct integer points with few ordinary lines."""
    # Try structured search first
    best = _brute_force_search(max_coord=30, time_limit=100)
    
    if best is not None:
        return best
    
    # Fallback: simple grid
    return [[x, y] for x in range(4) for y in range(3)]
# EVOLVE-BLOCK-END
