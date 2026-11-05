import time
import random
import numpy as np
from collections import defaultdict

_START = time.time()
_DEADLINE = 120.0   # evaluator kills at 150s; keep margin

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p, q):
    """Canonical integer line (a,b,c) for a*x + b*y = c through p,q, reduced."""
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = _gcd(_gcd(a, b), c)
    if g:
        a //= g; b //= g; c //= g
    # sign canonical: first non-zero of (a,b) positive
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _count4(pts):
    """Return (num_lines_exactly4, has_five_collinear) for list of 16 points."""
    n = len(pts)
    line_pts = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            line_pts[k].add(i)
            line_pts[k].add(j)
    cnt4 = 0
    has5 = False
    for k, s in line_pts.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
    return cnt4, has5

def _valid(pts):
    """Distinct, in range, no five collinear."""
    if len(set(map(tuple, pts))) != 16:
        return False
    for (x, y) in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, has5 = _count4(pts)
    return not has5

def _score(pts):
    if not _valid(pts):
        return -1
    c, _ = _count4(pts)
    return c

def _grid44():
    return [[x, y] for x in range(4) for y in range(4)]

def _random_start():
    pts = []
    seen = set()
    while len(pts) < 16:
        x = random.randint(-30, 30)
        y = random.randint(-30, 30)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        pts.append([x, y])
    return pts

def _neighbor(pts, rng):
    """Return a new point list with one point moved to a nearby random integer."""
    new = [list(p) for p in pts]
    idx = rng.randrange(16)
    base = new[idx]
    # try a few nearby moves with larger radius
    for _ in range(60):
        dx = rng.randint(-15, 15)
        dy = rng.randint(-15, 15)
        if dx == 0 and dy == 0:
            continue
        nx = base[0] + dx
        ny = base[1] + dy
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        cand = [list(p) for p in pts]
        cand[idx] = [nx, ny]
        if len(set(map(tuple, cand))) != 16:
            continue
        return cand
    return None

def _line_neighbor(pts, rng):
    """Move a point to lie on a line defined by two other points."""
    new = [list(p) for p in pts]
    idx = rng.randrange(16)
    # Pick two other points to define a line
    others = [i for i in range(16) if i != idx]
    i1, i2 = rng.sample(others, 2)
    p1, p2 = pts[i1], pts[i2]
    # Find integer points on the line through p1, p2 (excluding p1, p2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = _gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    step_x = dx // g
    step_y = dy // g
    # Generate candidate points on the line
    base_x, base_y = p1[0], p1[1]
    for t in range(-20, 21):
        if t == 0:
            continue
        nx = base_x + t * step_x
        ny = base_y + t * step_y
        if abs(nx) > 1000 or abs(ny) > 1000:
            continue
        cand = [list(p) for p in pts]
        cand[idx] = [nx, ny]
        if len(set(map(tuple, cand))) != 16:
            continue
        return cand
    return None

def _three_point_line_neighbor(pts, rng):
    """Move a point to complete a line with exactly 3 existing points to make 4."""
    n = 16
    idx = rng.randrange(n)
    # Find all lines with exactly 3 points (not including idx)
    others = [i for i in range(n) if i != idx]
    line_counts = defaultdict(list)
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            oi, oj = others[i], others[j]
            k = _line_key(pts[oi], pts[oj])
            line_counts[k].append((oi, oj))
    
    # Find lines with exactly 3 points among others
    candidate_lines = []
    for k, pairs in line_counts.items():
        point_set = set()
        for oi, oj in pairs:
            point_set.add(oi)
            point_set.add(oj)
        if len(point_set) == 3:
            candidate_lines.append((k, list(point_set)))
    
    if not candidate_lines:
        return None
    
    # Pick a random 3-point line
    k, three_pts = rng.choice(candidate_lines)
    
    # Get the line equation (a,b,c)
    a, b, c = k
    
    # Find integer points on this line that are not already in pts
    existing = set(tuple(p) for p in pts)
    # Parametrize the line
    if a == 0:
        # b*y = c => y = c/b (horizontal)
        if c % b != 0:
            return None
        y_val = c // b
        for x_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    elif b == 0:
        # a*x = c => x = c/a (vertical)
        if c % a != 0:
            return None
        x_val = c // a
        for y_val in range(-1000, 1001):
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    else:
        # a*x + b*y = c
        # Find one particular solution
        # Use extended gcd
        def _ext_gcd(a, b):
            if b == 0:
                return (1, 0, a)
            x1, y1, g = _ext_gcd(b, a % b)
            return (y1, x1 - (a // b) * y1, g)
        
        g = _gcd(a, b)
        if c % g != 0:
            return None
        a_div, b_div, c_div = a // g, b // g, c // g
        x0, y0, _ = _ext_gcd(a_div, b_div)
        x0 *= c_div
        y0 *= c_div
        
        # General solution: x = x0 + b_div*t, y = y0 - a_div*t
        for t in range(-50, 51):
            x_val = x0 + b_div * t
            y_val = y0 - a_div * t
            if abs(x_val) > 1000 or abs(y_val) > 1000:
                continue
            if (x_val, y_val) not in existing:
                cand = [list(p) for p in pts]
                cand[idx] = [x_val, y_val]
                return cand
    return None

def _local_search(start, rng, max_iters, deadline):
    cur = [list(p) for p in start]
    cur_s = _score(cur)
    if cur_s < 0:
        cur = _grid44()
        cur_s = _score(cur)
    best = [list(p) for p in cur]
    best_s = cur_s
    T = 1.0
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Alternate between different neighbor types
        r = rng.random()
        if r < 0.3:
            nb = _neighbor(cur, rng)
        elif r < 0.6:
            nb = _line_neighbor(cur, rng)
        else:
            nb = _three_point_line_neighbor(cur, rng)
        if nb is None:
            continue
        nb_s = _score(nb)
        if nb_s < 0:
            # invalid (five collinear or dup) -> reject
            continue
        if nb_s > cur_s:
            cur = nb
            cur_s = nb_s
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
        else:
            # simulated annealing acceptance
            d = nb_s - cur_s
            if d < 0 and rng.random() < np.exp(d / max(T, 1e-6)):
                cur = nb
                cur_s = nb_s
        T *= 0.999
    return best, best_s

def _try_affine_grid(rng, deadline):
    """Try various affine transformations of the 4x4 grid."""
    base = _grid44()
    best = base
    best_s = _score(base)
    
    # Try different affine transforms: (x,y) -> (ax+by+c, dx+ey+f)
    for a in range(-3, 4):
        for b in range(-3, 4):
            for d in range(-3, 4):
                for e in range(-3, 4):
                    if a * e - b * d == 0:
                        continue
                    for c in range(-5, 6):
                        for f in range(-5, 6):
                            pts = [[a * x + b * y + c, d * x + e * y + f] for x, y in base]
                            if len(set(map(tuple, pts))) != 16:
                                continue
                            if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                                continue
                            s = _score(pts)
                            if s > best_s:
                                best = pts
                                best_s = s
                    if time.time() > deadline:
                        return best, best_s
    return best, best_s

def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(12345)
    np.random.seed(12345)

    best = _grid44()
    best_s = _score(best)  # 10

    # Phase 1: Try affine transformations of the grid
    if time.time() < deadline - 10:
        b, s = _try_affine_grid(rng, deadline - 5)
        if s > best_s:
            best, best_s = b, s

    # Phase 2: local search from the best found
    if time.time() < deadline - 5:
        b, s = _local_search(best, rng, 30000, deadline - 2)
        if s > best_s:
            best, best_s = b, s

    # Phase 3: multiple random starts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        start = _random_start()
        b, s = _local_search(start, rng, 2000, deadline - 2)
        if s > best_s:
            best, best_s = b, s
        # occasionally re-seed from current best to refine
        if attempts % 3 == 0:
            b2, s2 = _local_search(best, rng, 2000, deadline - 2)
            if s2 > best_s:
                best, best_s = b2, s2

    # Final safety: ensure validity
    if not _valid(best):
        best = _grid44()
    return best
