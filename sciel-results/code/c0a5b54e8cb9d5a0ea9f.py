"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from collections import defaultdict

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    # Map from line (as normalized tuple of (dx,dy, intercept)) to count
    # Better: use pairs to define lines, then count points on each line
    line_points = defaultdict(list)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize the direction vector
            from math import gcd
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Ensure canonical direction (first non-zero component positive)
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            # Line: (dx, dy, c) where c = dy*x1 - dx*y1 (normalized)
            c = dy * x1 - dx * y1
            line_key = (dx, dy, c)
            line_points[line_key].append(i)
    
    count = 0
    for line_key, indices in line_points.items():
        if len(indices) == 4:
            count += 1
    
    return count

def _no_five_collinear(points):
    """Check that no line contains 5 or more points."""
    n = len(points)
    line_points = defaultdict(list)
    from math import gcd
    
    for i in range(n):
        for j in range(i+1, n):
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
            line_key = (dx, dy, c)
            line_points[line_key].append(i)
    
    for line_key, indices in line_points.items():
        if len(indices) >= 5:
            return False
    return True

def _valid(points):
    """Check all constraints: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if not _no_five_collinear(points):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    random.seed(42)
    
    # Start with 4x4 grid
    best_points = [[x, y] for x in range(4) for y in range(4)]
    best_score = _count_lines_exactly_4(best_points)
    
    # Try various initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Shifted grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4, 8)])
    
    # Various perturbations of grid
    for trial in range(20):
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(5):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-20, 20), random.randint(-20, 20)]
        if _valid(pts):
            initial_configs.append(pts)
    
    # Random configurations
    for trial in range(50):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-30, 30), random.randint(-30, 30)])
        if _valid(pts):
            initial_configs.append(pts)
    
    # Evaluate initial configs
    for cfg in initial_configs:
        if _valid(cfg):
            score = _count_lines_exactly_4(cfg)
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in cfg]
    
    # Local search: move one point at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            # Try moving point i to nearby positions
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    candidate = [p[:] for p in best_points]
                    candidate[i] = [new_x, new_y]
                    if _valid(candidate):
                        score = _count_lines_exactly_4(candidate)
                        if score > best_score:
                            best_score = score
                            best_points = candidate
                            improved = True
    
    # Second phase: try larger moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            for _ in range(100):
                new_x = random.randint(max(-1000, old_x - 20), min(1000, old_x + 20))
                new_y = random.randint(max(-1000, old_y - 20), min(1000, old_y + 20))
                candidate = [p[:] for p in best_points]
                candidate[i] = [new_x, new_y]
                if _valid(candidate):
                    score = _count_lines_exactly_4(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate
                        improved = True
    
    return best_points
# EVOLVE-BLOCK-END
