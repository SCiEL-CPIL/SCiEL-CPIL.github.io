import time
import math
import random
from collections import defaultdict

def _gcd3(a, b, c):
    g = math.gcd(a, b)
    g = math.gcd(g, c)
    return g

def count_ordinary_lines(points):
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        for j in range(i + 1, n):
            x2, y2 = points[j][0], points[j][1]
            dx = x2 - x1
            dy = y2 - y1
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            g = _gcd3(abs(a), abs(b), abs(c))
            if g:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a = -a
                b = -b
                c = -c
            lines[(a, b, c)].update((i, j))
    return sum(1 for s in lines.values() if len(s) == 2)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(map(tuple, points))) < 12:
        return False
    if any(abs(p[0]) > 10**9 or abs(p[1]) > 10**9 for p in points):
        return False
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    for k in range(3, 12):
        xi, yi = points[k]
        if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
            return True
    return False

def _build_grid_3x4():
    return [[x, y] for x in range(3) for y in range(4)]

def _build_pencil_4x3():
    # 4 lines, 3 points each
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pts = []
    for d in range(1, 4):
        for (ux, uy) in dirs:
            pts.append([d * ux, d * uy])
    return pts

def _build_pencil_4x3_shifted():
    # Shifted to avoid symmetry issues if any
    pts = _build_pencil_4x3()
    # Add offset
    return [[x + 10, y + 10] for x, y in pts]

def _build_random():
    rng = random.Random(42)
    pts = []
    while len(pts) < 12:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if [x, y] not in pts:
            pts.append([x, y])
    return pts

def construct():
    start = time.time()
    deadline = start + 140
    
    best_points = None
    best_score = float('inf')
    
    def consider(pts):
        nonlocal best_points, best_score
        if not _is_valid(pts):
            return
        s = count_ordinary_lines(pts)
        if s < best_score:
            best_score = s
            best_points = [[int(x), int(y)] for x, y in pts]

    # 1) Try diverse initial configurations
    initial_configs = [
        _build_grid_3x4(),
        _build_pencil_4x3(),
        _build_pencil_4x3_shifted(),
        _build_random()
    ]
    
    for cfg in initial_configs:
        consider(cfg)
        
    if best_score <= 6:
        return best_points

    # 2) Local Search
    rng = random.Random(7)
    cur = [p[:] for p in best_points]
    cur_score = best_score
    
    def try_move(moves):
        nonlocal cur, cur_score, best_points, best_score
        cand = [p[:] for p in cur]
        for idx, np in moves:
            cand[idx] = np
        if not _is_valid(cand):
            return
        s = count_ordinary_lines(cand)
        if s < cur_score:
            cur = cand
            cur_score = s
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in cand]

    iters = 0
    while time.time() < deadline and iters < 500000:
        iters += 1
        idx = rng.randint(0, 11)
        px, py = cur[idx]
        r = rng.random()
        
        if r < 0.5:
            # Snap onto a line through two other existing points
            a, b = rng.sample(range(12), 2)
            if a == idx or b == idx:
                continue
            p1, p2 = cur[a], cur[b]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            # Try integer points on this line
            # p = p1 + t * (dx, dy)
            # We want to move cur[idx] to a new point on this line
            # But cur[idx] might already be on it?
            # We just pick a random t
            t = rng.randint(-10, 10)
            if t == 0:
                continue
            nx = p1[0] + t * dx
            ny = p1[1] + t * dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                if [nx, ny] not in cur:
                    try_move([(idx, [nx, ny])])
        elif r < 0.8:
            # Small random step
            st = rng.choice([1, 2, 3, 5, 8, 13])
            nx = px + rng.randint(-st, st)
            ny = py + rng.randint(-st, st)
            if [nx, ny] not in cur:
                try_move([(idx, [nx, ny])])
        else:
            # Perturb two points
            i, j = rng.sample(range(12), 2)
            st = rng.choice([1, 2, 3, 5])
            m = [(i, [cur[i][0] + rng.randint(-st, st), cur[i][1] + rng.randint(-st, st)]),
                 (j, [cur[j][0] + rng.randint(-st, st), cur[j][1] + rng.randint(-st, st)])]
            try_move(m)

        if best_score <= 6:
            break

    if best_points is None:
        best_points = _build_grid_3x4()
    
    return best_points
