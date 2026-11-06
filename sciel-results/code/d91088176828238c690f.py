"""Rewritten program for minimizing ordinary lines among 12 integer points."""
import time
import random
from math import gcd
from itertools import combinations

DEADLINE = 145.0  # Safety margin before 150s limit

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_pairs = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(gcd(abs(a), abs(b)), abs(c)) if (a or b or c) else 1
            if g == 0:
                g = 1
            a, b, c = a // g, b // g, c // g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            line_pairs[(a, b, c)] = line_pairs.get((a, b, c), 0) + 1
    
    return sum(1 for cnt in line_pairs.values() if cnt == 1)

def _is_valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    # Not all collinear
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
        all_col = all((x2 - x1) * (yk - y1) - (y2 - y1) * (xk - x1) == 0 for xk, yk in points[3:])
        if all_col:
            return False
    return True

def _local_search(points, deadline, max_iter=50000):
    """Local search: try moving one point to a better position."""
    best = [p[:] for p in points]
    best_ordinary = _count_ordinary(best)
    n = 12
    iter_count = 0
    no_improve = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        idx = random.randint(0, n - 1)
        old = best[idx][:]
        
        # Try several candidate positions
        found_better = False
        for _ in range(20):
            # Mix of small and large moves
            if random.random() < 0.3:
                scale = random.choice([1, 2, 3, 5, 10])
            else:
                scale = random.choice([50, 100, 500, 1000, 10000])
            
            nx = old[0] + random.randint(-scale, scale)
            ny = old[1] + random.randint(-scale, scale)
            nx = max(-10**9, min(10**9, nx))
            ny = max(-10**9, min(10**9, ny))
            
            best[idx] = [nx, ny]
            if not _is_valid(best):
                best[idx] = old
                continue
            
            o = _count_ordinary(best)
            if o < best_ordinary:
                best_ordinary = o
                found_better = True
                no_improve = 0
                break
            else:
                best[idx] = old
        
        if not found_better:
            no_improve += 1
            if no_improve > 200:
                # Random restart
                for i in range(n):
                    best[i] = [random.randint(-50, 50), random.randint(-50, 50)]
                if _is_valid(best):
                    best_ordinary = _count_ordinary(best)
                no_improve = 0
    
    return best, best_ordinary

def construct():
    start = time.time()
    deadline = start + 145.0
    
    # Start with a 3x4 grid
    best_points = [[x, y] for x in range(4) for y in range(3)]
    best_ordinary = _count_ordinary(best_points)
    
    # Try several initial configurations
    initial_configs = [
        [[x, y] for x in range(4) for y in range(3)],  # 3x4 grid
        [[x, y] for x in range(6) for y in range(2)],  # 2x6 grid
        [[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[0,1],[1,1],[2,1],[3,1],[4,1],[5,1]],  # 2 rows of 6
        [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2],[0,3],[1,3],[2,3]],  # 3x4 vertical
        [[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],[0,2],[1,2],[2,2],[3,2]],  # 3x4
    ]
    
    for cfg in initial_configs:
        if time.time() > deadline - 30:
            break
        if not _is_valid(cfg):
            continue
        o = _count_ordinary(cfg)
        if o < best_ordinary:
            best_ordinary = o
            best_points = [p[:] for p in cfg]
    
    # Phase 1: Local search from best
    if time.time() < deadline - 20:
        best_points, best_ordinary = _local_search(best_points, deadline - 20, max_iter=30000)
    
    # Phase 2: Multiple restarts
    remaining = deadline - 25
    if remaining > 10:
        n_restarts = max(1, int(remaining / 8))
        for r in range(n_restarts):
            if time.time() > deadline - 5:
                break
            # Generate a random initial config in small range
            pts = []
            for _ in range(12):
                pts.append([random.randint(-20, 20), random.randint(-20, 20)])
            if not _is_valid(pts):
                continue
            o = _count_ordinary(pts)
            if o < best_ordinary:
                best_ordinary = o
                best_points = [p[:] for p in pts]
            
            # Local search from this config
            pts2, o2 = _local_search(pts, deadline - 5, max_iter=10000)
            if o2 < best_ordinary:
                best_ordinary = o2
                best_points = pts2
    
    # Final verification
    if not _is_valid(best_points):
        # Fallback to 3x4 grid
        best_points = [[x, y] for x in range(4) for y in range(3)]
    
    return [list(p) for p in best_points]
