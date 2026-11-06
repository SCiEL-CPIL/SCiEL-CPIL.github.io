import random
import time
import math
from collections import defaultdict

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            
            a = dy
            b = -dx
            c = a * p1[0] + b * p1[1]
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
    
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _make_initial_configs():
    configs = []
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    # 4x4 minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0,0],[3,0],[0,3],[3,3]]:
        pts.remove(corner)
    configs.append(pts)
    # 5x3 grid
    configs.append([[x, y] for x in range(5) for y in range(3)][:12])
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    # Random structured configs
    rng = random.Random(12345)
    for _ in range(30):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-6, 6), rng.randint(-6, 6)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)
    return configs

def construct():
    start_time = time.time()
    deadline = 140.0
    
    initial_configs = _make_initial_configs()
    
    best_points = None
    best_score = float('inf')
    
    for cfg in initial_configs:
        if len(cfg) != 12 or len(set(map(tuple, cfg))) < 12:
            continue
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = cfg[:]
        if best_score == 6:
            break
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)
    
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_p = best_points[idx][:]
        
        step_size = random.choice([1, 2, 3, 5, 8, 12, 16])
        dx = random.randint(-step_size, step_size)
        dy = random.randint(-step_size, step_size)
        if dx == 0 and dy == 0:
            continue
            
        new_p = [old_p[0] + dx, old_p[1] + dy]
        
        if abs(new_p[0]) > 10**9 or abs(new_p[1]) > 10**9:
            continue
            
        new_points = best_points[:]
        new_points[idx] = new_p
        if len(set(map(tuple, new_points))) < 12:
            continue
        
        score = count_ordinary_lines(new_points)
        
        if score < best_score:
            best_score = score
            best_points = new_points
            if best_score == 6:
                break
        
        # Occasional restart
        if random.random() < 0.02 and time.time() < deadline - 15:
            cfg = random.choice(initial_configs)
            if len(cfg) == 12 and len(set(map(tuple, cfg))) == 12:
                s = count_ordinary_lines(cfg)
                if s < best_score:
                    best_score = s
                    best_points = cfg[:]
    
    return best_points
