"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

def _line_key(p, q):
    """Canonical integer line key through points p and q: (a,b,c) with a*y - b*x = c,
    gcd(|a|,|b|)=1, sign fixed (a>0 or a==0 and b>0)."""
    x1, y1 = p
    x2, y2 = q
    a = y2 - y1
    b = x1 - x2
    g = math.gcd(abs(a), abs(b))
    if g:
        a //= g
        b //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * y1 - b * x1
    return (a, b, c)

def _score_fast(pts):
    """Return (num_lines_with_exactly_4_points, has_five_collinear).
    pts: list of 16 (x,y) ints. Fast O(n^2) with numpy for the 5-collinear check.
    """
    n = len(pts)
    # pair count per line (only need to detect lines with >=5 points and count ==4)
    pair_count = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            a = yj - yi
            b = xi - xj
            g = math.gcd(abs(a), abs(b))
            if g:
                a //= g
                b //= g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * yi - b * xi
            key = (a, b, c)
            pair_count[key] = pair_count.get(key, 0) + 1

    four = 0
    five = False
    for key, pc in pair_count.items():
        # k points on line => C(k,2) pairs
        if pc >= 10:  # C(5,2)=10
            five = True
        if pc == 6:   # C(4,2)=6
            four += 1
    return four, five

def _score(pts):
    four, five = _score_fast(pts)
    if five:
        return -1
    return four

def _validate(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    four, five = _score_fast(points)
    if five:
        return False
    return True

def _grid_config():
    return [[i, j] for i in range(4) for j in range(4)]

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _diverse_config():
    pts = []
    for _ in range(20000):
        if len(pts) >= 16:
            break
        p = _random_point()
        if p not in pts:
            pts.append(p)
    while len(pts) < 16:
        pts.append(_random_point())
    return pts[:16]

def _local_search(points, max_iters=4000, deadline=0.0):
    cur = [list(p) for p in points]
    cur_score = _score(cur)
    if cur_score < 0:
        return cur, -1
    best = [list(p) for p in cur]
    best_score = cur_score
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        idx = random.randint(0, 15)
        old = list(cur[idx])
        new = _random_point()
        if new == old:
            continue
        # distinctness
        dup = False
        for j, p in enumerate(cur):
            if j != idx and p == new:
                dup = True
                break
        if dup:
            continue
        cur[idx] = new
        ns = _score(cur)
        if ns > cur_score:
            cur_score = ns
            if ns > best_score:
                best_score = ns
                best = [list(p) for p in cur]
        else:
            cur[idx] = old
    return best, best_score

def _anneal(points, start_temp=800.0, end_temp=0.5, max_iters=6000, deadline=0.0):
    cur = [list(p) for p in points]
    cur_score = _score(cur)
    if cur_score < 0:
        return cur, -1
    best = [list(p) for p in cur]
    best_score = cur_score
    for it in range(max_iters):
        if time.time() >= deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        idx = random.randint(0, 15)
        old = list(cur[idx])
        new = _random_point()
        if new == old:
            continue
        dup = False
        for j, p in enumerate(cur):
            if j != idx and p == new:
                dup = True
                break
        if dup:
            continue
        cur[idx] = new
        ns = _score(cur)
        if ns >= cur_score:
            cur_score = ns
            if ns > best_score:
                best_score = ns
                best = [list(p) for p in cur]
        else:
            delta = ns - cur_score
            if random.random() < math.exp(delta / temp):
                cur_score = ns
            else:
                cur[idx] = old
    return best, best_score

def _polish(points, deadline=0.0, max_rounds=20):
    cur = [list(p) for p in points]
    cur_score = _score(cur)
    if cur_score < 0:
        return cur, -1
    best = [list(p) for p in cur]
    best_score = cur_score
    for r in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        for idx in range(16):
            if time.time() >= deadline:
                break
            ox, oy = cur[idx]
            for dx in range(-6, 7):
                for dy in range(-6, 7):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = ox + dx, oy + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    np_ = [nx, ny]
                    dup = False
                    for j, p in enumerate(cur):
                        if j != idx and p == np_:
                            dup = True
                            break
                    if dup:
                        continue
                    cur[idx] = np_
                    ns = _score(cur)
                    if ns > cur_score:
                        cur_score = ns
                        if ns > best_score:
                            best_score = ns
                            best = [list(p) for p in cur]
                        improved = True
                    else:
                        cur[idx] = [ox, oy]
        if not improved:
            break
    return best, best_score

def construct():
    start = time.time()
    deadline = start + 138  # safety margin under 150s

    best_points = _grid_config()
    best_score = _score(best_points)  # grid gives 20

    # Local search + anneal from the grid
    improved, s = _local_search(best_points, max_iters=3000, deadline=deadline)
    if s > best_score:
        best_score = s
        best_points = [list(p) for p in improved]

    improved, s = _anneal(best_points, start_temp=600.0, end_temp=0.5, max_iters=4000, deadline=deadline)
    if s > best_score:
        best_score = s
        best_points = [list(p) for p in improved]

    # Multiple random restarts
    for restart in range(8):
        if time.time() >= deadline:
            break
        if restart % 2 == 0:
            cfg = _diverse_config()
        else:
            # perturbed grid
            base = random.choice([3, 4, 5])
            off = random.randint(-2, 2)
            cfg = [[i + off, j + off] for i in range(base) for j in range(base)]
            while len(cfg) < 16:
                p = _random_point()
                if p not in cfg:
                    cfg.append(p)
            cfg = cfg[:16]

        improved, s = _local_search(cfg, max_iters=1500, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in improved]

        improved, s = _anneal(improved, start_temp=500.0, end_temp=0.5, max_iters=2500, deadline=deadline)
        if s > best_score:
            best_score = s
            best_points = [list(p) for p in improved]

    # Final polish
    polished, s = _polish(best_points, deadline=deadline, max_rounds=15)
    if s > best_score:
        best_score = s
        best_points = [list(p) for p in polished]

    if best_points is None or not _validate(best_points):
        return _grid_config()
    return best_points
# EVOLVE-BLOCK-END
