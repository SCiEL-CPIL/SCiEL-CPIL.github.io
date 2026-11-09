import time
import math

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _known_optimal():
    # Hexagon of radius 3 has 37 points. Remove 5 lowest-degree boundary points.
    pts = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if _hexdist(a,b) <= 3:
                pts.append((a,b))
    s = set(pts)
    deg = {}
    for p in pts:
        d = 0
        for da,db in _NEIGH:
            if (p[0]+da, p[1]+db) in s:
                d += 1
        deg[p] = d
    pts_sorted = sorted(pts, key=lambda p: (deg[p], _hexdist(p[0],p[1])))
    to_remove = set(pts_sorted[:5])
    result = [p for p in pts if p not in to_remove]
    return result

def construct():
    deadline = time.time() + 110
    n = 32

    # Start from a near-optimal hexagonal configuration
    base = _known_optimal()
    best = list(base)
    best_e = _count_edges(best)

    # If already optimal, return immediately
    if best_e >= 76:
        return [[a,b] for (a,b) in best]

    # Local search to try to reach 76
    import random
    rng = random.Random(0)
    in_set = set(best)
    pool = set()
    for (a,b) in best:
        for r in range(0,4):
            for a2 in range(a-r, a+r+1):
                for b2 in range(b-r, b+r+1):
                    if _hexdist(a2,b2) <= 8:
                        pool.add((a2,b2))
    pool = list(pool)
    it = 0
    while time.time() < deadline and it < 200000:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        trial = best[:i] + [cand] + best[i+1:]
        e2 = _count_edges(trial)
        if e2 > best_e:
            best_e = e2
            best = trial
            in_set.discard(cur)
            in_set.add(cand)
        if best_e >= 76:
            break
    return [[a,b] for (a,b) in best]
