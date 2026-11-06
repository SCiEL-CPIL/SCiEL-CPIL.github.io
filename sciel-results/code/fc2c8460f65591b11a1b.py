"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from collections import defaultdict

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    if A == 0 and B == 0:
        return None
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 1:
        A //= g
        B //= g
        C //= g
    return (A, B, C)

def _count_ordinary(points):
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].add(i)
            line_map[key].add(j)
    return sum(1 for s in line_map.values() if len(s) == 2)

def _all_collinear(pts):
    if len(pts) < 3:
        return False
    (x0,y0),(x1,y1),(x2,y2) = pts[0],pts[1],pts[2]
    if (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0) != 0:
        return False
    for i in range(3, len(pts)):
        xi, yi = pts[i]
        if (x1-x0)*(yi-y0) - (y1-y0)*(xi-x0) != 0:
            return False
    return True

def _local_search(points, time_budget=10.0):
    current = [p[:] for p in points]
    current_score = _count_ordinary(current)
    n = len(current)
    start_time = time.time()
    
    def build_candidates():
        cands = set()
        for p in current:
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    cands.add((p[0]+dx, p[1]+dy))
        for _ in range(20):
            cands.add((random.randint(-20,20), random.randint(-20,20)))
        return list(cands)
    
    candidates = build_candidates()
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        best_move = None
        best_score = current_score
        for i in range(n):
            orig = tuple(current[i])
            for pos in candidates:
                if pos == orig:
                    continue
                x, y = pos
                if abs(x) > 10**9 or abs(y) > 10**9:
                    continue
                dup = False
                for j in range(n):
                    if j != i and current[j][0] == x and current[j][1] == y:
                        dup = True
                        break
                if dup:
                    continue
                new_pts = [p[:] for p in current]
                new_pts[i] = [x, y]
                if _all_collinear(new_pts):
                    continue
                score = _count_ordinary(new_pts)
                if score < best_score:
                    best_score = score
                    best_move = (i, [x, y])
        if best_move is not None:
            i, new_pos = best_move
            current[i] = new_pos
            current_score = best_score
            improved = True
            candidates = build_candidates()
    
    return current, current_score

def construct():
    deadline = time.time() + 140
    random.seed(777)
    best_points = None
    best_score = float('inf')
    
    configs = []
    for r, c in [(3,4),(4,3),(2,6),(6,2)]:
        pts = [[j, i] for i in range(r) for j in range(c)]
        configs.append(pts)
    for _ in range(25):
        r = random.choice([2,3,4])
        c = 12 // r
        if r * c != 12:
            continue
        base = [[j*2, i*2] for i in range(r) for j in range(c)]
        pts = [p[:] for p in base]
        for p in pts:
            p[0] += random.randint(-4,4)
            p[1] += random.randint(-4,4)
        configs.append(pts)
    for _ in range(15):
        pts = [[random.randint(-15,15), random.randint(-15,15)] for _ in range(12)]
        configs.append(pts)
    
    for config in configs:
        if time.time() > deadline:
            break
        seen = set()
        valid = True
        for p in config:
            t = tuple(p)
            if t in seen or abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
                valid = False
                break
            seen.add(t)
        if not valid or len(seen) != 12:
            continue
        if _all_collinear(config):
            continue
        pts, score = _local_search(config, time_budget=8.0)
        if score < best_score:
            best_score = score
            best_points = pts
            if best_score <= 6:
                break
    
    if best_points is None:
        return [[x, y] for x in range(4) for y in range(3)]
    return best_points
# EVOLVE-BLOCK-END
