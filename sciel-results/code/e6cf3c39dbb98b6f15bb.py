import math
import itertools

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

def construct():
    n = 32
    
    # Generate points in spiral order (by hex distance, then angle)
    # We need enough points to pick 32. Radius 3 gives 37, which is enough.
    R = 4
    pool = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                pool.append((a,b))
                
    # Sort by hex distance, then angle
    def _sort_key(p):
        a, b = p
        hd = _hexdist(a,b)
        x = a + 0.5*b
        y = 0.8660254037844386*b
        ang = math.atan2(y, x)
        return (hd, ang)
        
    pool.sort(key=_sort_key)
    
    # Take the first 32 points
    best = pool[:n]
    best_e = _count_edges(best)
    
    # If we haven't reached the optimum, run a local search to improve
    if best_e < 76:
        import random
        import time
        deadline = time.time() + 100
        rng = random.Random(0)
        in_set = set(best)
        
        it = 0
        while time.time() < deadline and it < 500000:
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
            if best_e == 76:
                break
    
    return [[a,b] for (a,b) in best]
