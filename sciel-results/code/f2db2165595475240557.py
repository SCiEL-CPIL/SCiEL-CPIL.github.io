import time
from math import gcd
import random

def _count_ordinary_fast(points):
    n = len(points)
    line_to_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            dx //= g
            dy //= g
            c = -(dx * y1 + dy * x1)
            g2 = gcd(gcd(abs(dx), abs(dy)), abs(c))
            if g2 > 0:
                dx //= g2
                dy //= g2
                c //= g2
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy, c = -dx, -dy, -c
            key = (dx, dy, c)
            if key not in line_to_pts:
                line_to_pts[key] = set()
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    ordinary = sum(1 for pts in line_to_pts.values() if len(pts) == 2)
    return ordinary

def _valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
    x0, y0 = points[0]
    x1, y1 = points[1]
    dx = x1 - x0
    dy = y1 - y0
    for x, y in points[2:]:
        if dx * (y - y0) != dy * (x - x0):
            return True
    return False

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    random.seed(42)
    
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_score = _count_ordinary_fast(best_points)
    
    def try_config(pts):
        nonlocal best_points, best_score
        if not _valid(pts):
            return
        s = _count_ordinary_fast(pts)
        if s < best_score:
            best_score = s
            best_points = [list(p) for p in pts]
    
    # Try many structured configs
    configs = []
    # Grids
    for w in range(2, 7):
        for h in range(2, 7):
            if w * h >= 12:
                pts = [(x, y) for x in range(w) for y in range(h)][:12]
                if len(pts) == 12:
                    configs.append(pts)
    # Add all
    for cfg in configs:
        try_config(cfg)
    
    # Local search
    current = [list(p) for p in best_points]
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        if best_score <= 6:
            break
        
        i = random.randint(0, 11)
        x, y = current[i]
        
        best_trial = None
        best_trial_score = best_score
        
        # Try nearby positions
        for dx in range(-10, 11):
            for dy in range(-10, 11):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                trial = [list(p) for p in current]
                trial[i] = [nx, ny]
                if len(set(tuple(p) for p in trial)) != 12:
                    continue
                if not _valid(trial):
                    continue
                s = _count_ordinary_fast(trial)
                if s < best_trial_score:
                    best_trial_score = s
                    best_trial = trial
        
        # Try some random positions
        for _ in range(3):
            nx, ny = random.randint(-30, 30), random.randint(-30, 30)
            if abs(nx) > 10**9 or abs(ny) > 10**9:
                continue
            trial = [list(p) for p in current]
            trial[i] = [nx, ny]
            if len(set(tuple(p) for p in trial)) != 12:
                continue
            if not _valid(trial):
                continue
            s = _count_ordinary_fast(trial)
            if s < best_trial_score:
                best_trial_score = s
                best_trial = trial
        
        if best_trial is not None:
            current = best_trial
            best_score = best_trial_score
            best_points = [list(p) for p in current]
    
    return [[p[0], p[1]] for p in best_points]
