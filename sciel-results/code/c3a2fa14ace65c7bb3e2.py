import time
import random
import math
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x1 * y2 - x2 * y1
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return None
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _count_ordinary_fast(points):
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = defaultdict(int)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key is not None:
                line_counts[key] += 1
    
    ordinary_count = 0
    for count in line_counts.values():
        if count == 2:
            ordinary_count += 1
    return ordinary_count

def _fast_local_search(points, deadline):
    current = [p[:] for p in points]
    current_score = _count_ordinary_fast(current)
    best = [p[:] for p in current]
    best_score = current_score
    n = len(current)
    
    while time.time() < deadline:
        improved = False
        # Try moving each point to a nearby location
        for idx in range(n):
            if time.time() > deadline:
                break
            ox, oy = current[idx]
            # Small neighborhood
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 10**9 or abs(ny) > 10**9:
                        continue
                    # Check distinct
                    if any(current[k][0] == nx and current[k][1] == ny for k in range(n) if k != idx):
                        continue
                    new_points = [p[:] for p in current]
                    new_points[idx] = [nx, ny]
                    new_score = _count_ordinary_fast(new_points)
                    if new_score < current_score:
                        current = new_points
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best = [p[:] for p in current]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Random jump
            idx = random.randint(0, n - 1)
            nx = random.randint(-10, 10)
            ny = random.randint(-10, 10)
            if not any(current[k][0] == nx and current[k][1] == ny for k in range(n) if k != idx):
                new_points = [p[:] for p in current]
                new_points[idx] = [nx, ny]
                new_score = _count_ordinary_fast(new_points)
                if new_score < current_score + 1: # Accept slightly worse to escape local minima
                    current = new_points
                    current_score = new_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial configurations
    configs = []
    # 3x4 Grid
    pts = [[j, i] for i in range(3) for j in range(4)]
    configs.append(pts)
    
    # 2x6 Grid
    pts = [[j, i] for i in range(2) for j in range(6)]
    configs.append(pts)
    
    # Random
    for _ in range(5):
        pts = []
        seen = set()
        while len(pts) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        configs.append(pts)
    
    best_overall = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() > deadline:
            break
        # Allocate remaining time / remaining configs
        time_left = deadline - time.time()
        per_config_time = time_left / max(1, len(configs))
        search_deadline = time.time() + min(10, per_config_time)
        
        result, score = _fast_local_search(config, search_deadline)
        if score < best_score:
            best_score = score
            best_overall = result
            
        if best_score <= 6:
            break
            
    if best_overall is None:
        best_overall = configs[0]
        
    return best_overall
