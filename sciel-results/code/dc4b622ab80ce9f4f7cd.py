"""Sixteen integer points, many 4-point lines, no five collinear."""
import time
import random
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s wall-clock

def _line_key(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        return (0, 1, p1[0], 0)
    if dy == 0:
        return (1, 0, 0, p1[1])
    g = abs(dx)
    while dy % g != 0:
        g = abs(g % dy) if g else abs(dy)
        if g == 0:
            g = abs(dx)
            break
    # g = gcd(|dx|,|dy|)
    import math
    g = math.gcd(abs(dx), abs(dy))
    ux, uy = dx // g, dy // g
    if (ux < 0) or (ux == 0 and uy < 0):
        ux, uy = -ux, -uy
    # canonical point: min of the two along direction
    if (p1[0] * uy - p1[1] * ux) > (p2[0] * uy - p2[1] * ux):
        cx, cy = p2[0], p2[1]
    else:
        cx, cy = p1[0], p1[1]
    return (ux, uy, cx, cy)

def _count4(points):
    """Return (score4, max_collinear) for a list of 16 [x,y] points."""
    n = len(points)
    if n < 4:
        return 0, 0
    keys = []
    for i in range(n):
        for j in range(i + 1, n):
            keys.append(_line_key(points[i], points[j]))
    uniq = list(set(keys))
    if not uniq:
        return 0, 0
    idx = {k: i for i, k in enumerate(uniq)}
    m = len(uniq)
    cnt = np.bincount([idx[k] for k in keys], minlength=m)
    maxc = int(cnt.max())
    score = int(np.sum(cnt == 4))
    return score, maxc

def _valid(points):
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if not (isinstance(p, list) and len(p) == 2):
            return False
        x, y = p
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if abs(x) > 1000 or abs(y) > 1000:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    _, maxc = _count4(points)
    return maxc < 5

def _rand_pt(rng):
    return [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]

def _rand_cfg(rng):
    while True:
        pts = [_rand_pt(rng) for _ in range(16)]
        if _valid(pts):
            return pts

def _grid_variant(rng):
    # 4x4 grid with random integer affine transform (det != 0), then jitter-free
    # Choose base 4x4 grid, apply random invertible integer affine map.
    while True:
        a = rng.randint(-20, 20)
        b = rng.randint(-20, 20)
        c = rng.randint(-20, 20)
        d = rng.randint(-20, 20)
        if a * d - b * c == 0:
            continue
        ox = rng.randint(-500, 500)
        oy = rng.randint(-500, 500)
        pts = []
        ok = True
        seen = set()
        for x in range(4):
            for y in range(4):
                nx = a * x + b * y + ox
                ny = c * x + d * y + oy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    ok = False
                    break
                if (nx, ny) in seen:
                    ok = False
                    break
                seen.add((nx, ny))
                pts.append([nx, ny])
            if not ok:
                break
        if ok and _valid(pts):
            return pts

def _crossover(p1, p2, rng):
    # uniform crossover
    child = []
    for i in range(16):
        child.append(list(p1[i] if rng.random() < 0.5 else p2[i]))
    return child

def _mutate(pt, rng, step):
    x = pt[0] + rng.randint(-step, step)
    y = pt[1] + rng.randint(-step, step)
    x = max(-1000, min(1000, x))
    y = max(-1000, min(1000, y))
    return [x, y]

def construct():
    rng = random.Random(12345)
    # seed population
    pop = []
    # base grid
    base = [[x, y] for x in range(4) for y in range(4)]
    pop.append(base)
    for _ in range(5):
        try:
            pop.append(_grid_variant(rng))
        except Exception:
            pass
    for _ in range(20):
        try:
            pop.append(_rand_cfg(rng))
        except Exception:
            pass
    # dedupe
    uniq = []
    seen = set()
    for p in pop:
        k = tuple(tuple(q) for q in p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    pop = uniq

    # evaluate
    scored = []
    for p in pop:
        s, mc = _count4(p)
        scored.append((s, mc, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    best = scored[0][2]
    best_score = scored[0][0]

    # hill climb on each of top configs
    def local_search(cfg, iters):
        nonlocal best, best_score
        cur = [list(p) for p in cfg]
        s, mc = _count4(cur)
        for _ in range(iters):
            if time.time() > _DEADLINE:
                break
            i = rng.randrange(16)
            step = rng.choice([1, 2, 3, 5, 8, 13, 20, 40])
            old = cur[i]
            cur[i] = _mutate(old, rng, step)
            if not _valid(cur):
                cur[i] = old
                continue
            ns, nmc = _count4(cur)
            if ns > s or (ns == s and nmc < mc):
                s, mc = ns, nmc
                if s > best_score:
                    best_score = s
                    best = [list(p) for p in cur]
            else:
                cur[i] = old
        return cur, s, mc

    # iterate local search over population
    top = scored[:min(12, len(scored))]
    for s, mc, cfg in top:
        if time.time() > _DEADLINE:
            break
        local_search(cfg, 400)

    # evolutionary loop
    pop2 = [[list(p) for p in cfg] for _, _, cfg in top]
    while time.time() < _DEADLINE - 2.0:
        # breed
        a, b = rng.sample(pop2, 2)
        child = _crossover(a, b, rng)
        if not _valid(child):
            continue
        s, mc = _count4(child)
        if s > best_score:
            best_score = s
            best = [list(p) for p in child]
        # local search on child
        child, s, mc = local_search(child, 150)
        # keep best in pop
        pop2.append(child)
        # trim
        scored2 = []
        for p in pop2:
            ss, mcc = _count4(p)
            scored2.append((ss, mcc, p))
        scored2.sort(key=lambda t: (-t[0], t[1]))
        pop2 = [[list(p) for p in cfg] for _, _, cfg in scored2[:16]]

    # final safety: ensure best is valid
    if not _valid(best):
        # fallback to a known-valid grid
        best = [[x, y] for x in range(4) for y in range(4)]
    return best
