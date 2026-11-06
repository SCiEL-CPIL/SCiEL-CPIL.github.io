import time
import random
from itertools import combinations
from math import gcd

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 3:
        return 0
    line_counts = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx_n = dx // g
            dy_n = dy // g
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
            if dx_n == 0:
                key = ('v', x1)
            else:
                c = dy_n * x1 - dx_n * y1
                key = ('s', dx_n, dy_n, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    ordinary = 0
    for key, cnt in line_counts.items():
        if cnt == 1:
            ordinary += 1
    return ordinary

def _not_all_collinear(points):
    if len(points) < 3:
        return True
    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    for i in range(2, len(points)):
        x, y = points[i]
        if dx * (y - y1) != dy * (x - x1):
            return True
    return False

def _local_search(points, deadline):
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        for idx in range(12):
            old_x, old_y = current[idx]
            # Try nearby moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    is_dup = False
                    for j in range(12):
                        if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                    current[idx] = [new_x, new_y]
                    if not _not_all_collinear(current):
                        current[idx] = [old_x, old_y]
                        continue
                    new_score = _count_ordinary(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best = [p[:] for p in current]
                        break
                    else:
                        current[idx] = [old_x, old_y]
                else:
                    continue
                break
            if improved:
                break
        if not improved:
            # Random restart within neighborhood
            idx = random.randint(0, 11)
            old_x, old_y = current[idx]
            for _ in range(50):
                new_x = old_x + random.randint(-10, 10)
                new_y = old_y + random.randint(-10, 10)
                if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                    continue
                is_dup = False
                for j in range(12):
                    if j != idx and current[j][0] == new_x and current[j][1] == new_y:
                        is_dup = True
                        break
                if is_dup:
                    continue
                current[idx] = [new_x, new_y]
                if not _not_all_collinear(current):
                    current[idx] = [old_x, old_y]
                    continue
                new_score = _count_ordinary(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best = [p[:] for p in current]
                    break
                else:
                    current[idx] = [old_x, old_y]
            if not improved:
                break
    
    return best, best_score

def construct():
    deadline = time.time() + 130
    
    random.seed(12345)
    
    # Multiple initial configurations
    initial_configs = []
    
    # 3x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 4x3 grid
    initial_configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # Structured: two rows of 6
    initial_configs.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])
    
    # Structured: 3 rows of 4
    initial_configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # Random configs
    for _ in range(20):
        pts = []
        while len(pts) < 12:
            x = random.randint(-15, 15)
            y = random.randint(-15, 15)
            if [x, y] not in pts:
                pts.append([x, y])
        if _not_all_collinear(pts):
            initial_configs.append(pts)
    
    best_config = None
    best_score = float('inf')
    
    for cfg in initial_configs:
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_config = [p[:] for p in cfg]
    
    # Local search from best initial
    if best_config:
        best_config, best_score = _local_search(best_config, deadline)
    
    # Try local search from other good configs if time permits
    for cfg in initial_configs:
        if time.time() > deadline:
            break
        score = _count_ordinary(cfg)
        if score <= best_score + 5:  # Only try promising ones
            cfg_best, cfg_score = _local_search(cfg, deadline)
            if cfg_score < best_score:
                best_score = cfg_score
                best_config = cfg_best
    
    return best_config
