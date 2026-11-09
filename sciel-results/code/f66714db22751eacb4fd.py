import time
import math
import random

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

def _greedy_construct(rng, n=32):
    """Build set incrementally, adding point that maximizes new edges."""
    in_set = set()
    pts = []
    # Candidate pool: all lattice points within hex radius 6
    pool = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            if _hexdist(a,b) <= 6:
                pool.append((a,b))
    rng.shuffle(pool)
    
    for _ in range(n):
        best_pt = None
        best_gain = -1
        for cand in pool:
            if cand in in_set:
                continue
            # Count how many neighbors cand has in in_set
            gain = 0
            for da, db in _NEIGH:
                if (cand[0]+da, cand[1]+db) in in_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_pt = cand
            elif gain == best_gain and best_pt is not None:
                # Tie-break: prefer closer to centroid
                if _hexdist(cand[0], cand[1]) < _hexdist(best_pt[0], best_pt[1]):
                    best_pt = cand
        if best_pt is None:
            break
        in_set.add(best_pt)
        pts.append(best_pt)
    return pts

def _local_search(pts, deadline, rng, n=32):
    """Incremental local search: swap one point for another if it improves edge count."""
    best = list(pts)
    in_set = set(best)
    best_e = _count_edges(best)
    
    if best_e >= 76:
        return best, best_e
    
    # Build candidate pool
    pool = set()
    for (a,b) in best:
        for da in range(-4, 5):
            for db in range(-4, 5):
                na, nb = a+da, b+db
                if _hexdist(na, nb) <= 8:
                    pool.add((na, nb))
    pool = list(pool)
    
    it = 0
    while time.time() < deadline:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        # Incremental edge count: remove cur, add cand
        # Edges lost: neighbors of cur that are in in_set (excluding cur itself)
        lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                lost += 1
        
        # Edges gained: neighbors of cand that are in in_set (excluding cur, since cur is being removed)
        gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                gained += 1
        
        delta = gained - lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            if best_e >= 76:
                break
        elif delta == 0 and rng.random() < 0.1:
            # Accept neutral moves sometimes to escape local optima
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
    
    return best, best_e

def construct():
    deadline = time.time() + 110
    n = 32
    
    best_overall = None
    best_e_overall = 0
    
    # Strategy 1: Start from hexagonal base
    base = _known_optimal()
    rng1 = random.Random(42)
    pts1, e1 = _local_search(base, deadline, rng1, n)
    if e1 > best_e_overall:
        best_e_overall = e1
        best_overall = pts1
    if best_e_overall >= 76:
        return [[a,b] for (a,b) in best_overall]
    
    # Strategy 2: Greedy construction with different seeds
    for seed in [0, 1, 2, 3]:
        if time.time() > deadline:
            break
        rng2 = random.Random(seed)
        pts2 = _greedy_construct(rng2, n)
        if len(pts2) < n:
            continue
        e2 = _count_edges(pts2)
        if e2 > best_e_overall:
            best_e_overall = e2
            best_overall = pts2
            if best_e_overall >= 76:
                return [[a,b] for (a,b) in best_overall]
        # Local search from greedy result
        remaining = deadline - time.time()
        if remaining > 2:
            rng3 = random.Random(seed + 100)
            pts3, e3 = _local_search(pts2, deadline, rng3, n)
            if e3 > best_e_overall:
                best_e_overall = e3
                best_overall = pts3
                if best_e_overall >= 76:
                    return [[a,b] for (a,b) in best_overall]
    
    return [[a,b] for (a,b) in best_overall]
