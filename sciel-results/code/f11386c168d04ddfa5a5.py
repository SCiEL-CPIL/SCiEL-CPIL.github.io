import time
import random
import math
from math import gcd
from collections import defaultdict

def _count_lines_exactly_4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    if n < 4:
        return 0
    line_points = defaultdict(set)
    
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
            line_key = (dx, dy, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
    count = 0
    for line_key, indices in line_points.items():
        if len(indices) == 4:
            count += 1
    return count

def _no_five_collinear(points):
    """Check that no line contains 5 or more points."""
    n = len(points)
    line_points = defaultdict(set)
    
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
            line_key = (dx, dy, c)
            line_points[line_key].add(i)
            line_points[line_key].add(j)
    
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

def _evaluate(points):
    """Return score if valid, else -1."""
    if not _valid(points):
        return -1
    return _count_lines_exactly_4(points)

def _local_search(points, deadline, max_iter=200):
    """Hill-climbing local search with small moves."""
    current = [p[:] for p in points]
    current_score = _evaluate(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    iters = 0
    while improved and iters < max_iter and time.time() < deadline:
        improved = False
        iters += 1
        for i in range(16):
            if time.time() > deadline:
                return current, current_score
            ox, oy = current[i]
            # Try all offsets in a small neighborhood
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    candidate = [p[:] for p in current]
                    candidate[i] = [nx, ny]
                    score = _evaluate(candidate)
                    if score > current_score:
                        current = candidate
                        current_score = score
                        improved = True
                        break
                if improved:
                    break
    return current, current_score

def _large_move_search(points, deadline, n_trials=500):
    """Local search with larger random moves to escape local optima."""
    current = [p[:] for p in points]
    current_score = _evaluate(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    trials = 0
    while improved and trials < n_trials and time.time() < deadline:
        improved = False
        trials += 1
        for i in range(16):
            if time.time() > deadline:
                return current, current_score
            ox, oy = current[i]
            for _ in range(50):
                nx = random.randint(max(-1000, ox - 30), min(1000, ox + 30))
                ny = random.randint(max(-1000, oy - 30), min(1000, oy + 30))
                candidate = [p[:] for p in current]
                candidate[i] = [nx, ny]
                score = _evaluate(candidate)
                if score > current_score:
                    current = candidate
                    current_score = score
                    improved = True
                    break
            if improved:
                break
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Standard 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 2. Shifted 4x4 grid
    initial_configs.append([[x + 1, y + 1] for x in range(4) for y in range(4)])
    
    # 3. Scaled grid
    initial_configs.append([[x * 2, y * 2] for x in range(4) for y in range(4)])
    
    # 4. Perturbed grids
    for trial in range(30):
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(random.randint(2, 6)):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-10, 10), random.randint(-10, 10)]
        if _valid(pts):
            initial_configs.append(pts)
    
    # 5. Random configurations
    for trial in range(80):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-20, 20), random.randint(-20, 20)])
        if _valid(pts):
            initial_configs.append(pts)
    
    # 6. Structured alternatives: two rows of 8 with offsets
    for offset in range(0, 10):
        pts = [[x, 0] for x in range(8)] + [[x + offset, 3] for x in range(8)]
        if _valid(pts):
            initial_configs.append(pts)
    
    # Evaluate all initial configs
    best_points = None
    best_score = -1
    for cfg in initial_configs:
        if _valid(cfg):
            score = _count_lines_exactly_4(cfg)
            if score > best_score:
                best_score = score
                best_points = [p[:] for p in cfg]
    
    if best_points is None:
        # Fallback: return the 4x4 grid
        return [[x, y] for x in range(4) for y in range(4)]
    
    # Phase 1: Fine-grained local search from best initial
    improved_points, score = _local_search(best_points, deadline, max_iter=300)
    if score > best_score:
        best_score = score
        best_points = improved_points
    
    # Phase 2: Large move search to escape local optima
    improved_points, score = _large_move_search(best_points, deadline, n_trials=800)
    if score > best_score:
        best_score = score
        best_points = improved_points
    
    # Phase 3: Another fine-grained polish
    improved_points, score = _local_search(best_points, deadline, max_iter=200)
    if score > best_score:
        best_score = score
        best_points = improved_points
    
    # Phase 4: Try multi-start from different initial configs
    for cfg in initial_configs[:20]:
        if time.time() > deadline:
            break
        if _valid(cfg):
            improved_points, score = _local_search(cfg, deadline, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = improved_points
    
    # Final polish
    improved_points, score = _local_search(best_points, deadline, max_iter=100)
    if score > best_score:
        best_score = score
        best_points = improved_points
    
    return best_points
