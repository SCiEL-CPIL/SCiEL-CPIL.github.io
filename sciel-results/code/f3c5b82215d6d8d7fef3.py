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

def _spiral_order(limit):
    R = limit
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            hd = _hexdist(a,b)
            x = a + 0.5*b
            y = 0.8660254037844386*b
            ang = math.atan2(y, x)
            cands.append((hd, ang, a, b))
    cands.sort()
    out = []
    for hd, ang, a, b in cands:
        out.append((a,b))
    return out

def _is_corner_r3(a, b):
    # R3 points are those with hexdist 3
    # Corners have 1 neighbor in R2, Edges have 2.
    # Corners are (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)
    # Check if it is one of these
    return (a,b) in [(3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3)]

def construct():
    deadline = time.time() + 120
    n = 32
    order = _spiral_order(12)
    
    # Base: first 19 points (R<=2)
    base = order[:19]
    base_set = set(base)
    
    # Candidates for the next 13 points: R3 points
    r3_points = [p for p in order if _hexdist(p[0], p[1]) == 3]
    
    # Sort R3 points by angle (already done in order)
    # We want to pick 13 points that minimize the number of corners.
    # There are 6 corners and 12 edges in R3.
    # Any 13 contiguous points will have at least 4 corners?
    # Let's just pick the 13 points with the fewest corners.
    # Since they are on a ring, we can slide a window of 13.
    
    # r3_points is sorted by angle.
    # Duplicate to handle wrap-around
    r3_ext = r3_points + r3_points
    best_subset = None
    min_corners = 100
    
    for i in range(len(r3_points)):
        window = r3_ext[i:i+13]
        corners = sum(1 for p in window if _is_corner_r3(p[0], p[1]))
        if corners < min_corners:
            min_corners = corners
            best_subset = window
    
    # Combine
    best = base + best_subset
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Local search to ensure optimality (in case of ties or errors)
    pool_set = set()
    for r in range(10):
        for a in range(-r, r+1):
            for b in range(-r, r+1):
                if _hexdist(a,b) == r:
                    pool_set.add((a,b))
    pool = list(pool_set)

    rng = random.Random(0)
    it = 0
    no_improve = 0
    max_no_improve = 5000

    while time.time() < deadline and no_improve < max_no_improve:
        it += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            no_improve += 1
            continue

        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                edges_lost += 1
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                edges_gained += 1

        delta = edges_gained - edges_lost
        if delta > 0:
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            no_improve = 0
        else:
            no_improve += 1

    return [[a,b] for (a,b) in best]
