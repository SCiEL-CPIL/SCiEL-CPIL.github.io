import time
import random
import math

_START = None
_DEADLINE = None

def _now():
    return time.time()

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _hexdist(p, q):
    da = p[0] - q[0]
    db = p[1] - q[1]
    return (abs(da) + abs(da + db) + abs(db)) // 2

def _edge_count(Sset):
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    if len(Sset) != 32:
        return False
    return all(abs(a) <= 64 and abs(b) <= 64 for a, b in Sset)

def _fast_local_search(Sset, pool, swap_cand, deadline, rng, max_passes=2000):
    """Optimized local search: 1-swap + 2-swap moves."""
    S = list(Sset)
    Sset = set(S)
    cur = _edge_count(Sset)
    
    for _ in range(max_passes):
        if _now() > deadline - 1.0:
            break
        improved = False
        
        # Pass 1: 1-swap moves
        order = list(range(len(S)))
        rng.shuffle(order)
        for i in order:
            p = S[i]
            for q in swap_cand[p]:
                if q in Sset:
                    continue
                newS = set(Sset)
                newS.discard(p)
                newS.add(q)
                ne = _edge_count(newS)
                if ne > cur:
                    Sset = newS
                    S[i] = q
                    cur = ne
                    improved = True
                    break
            if improved:
                break
        
        # Pass 2: 2-swap moves (only if 1-swap found nothing)
        if not improved:
            order2 = list(range(len(S)))
            rng.shuffle(order2)
            for i in order2:
                if _now() > deadline - 1.0:
                    break
                p1 = S[i]
                for j in range(i + 1, len(S)):
                    if _now() > deadline - 1.0:
                        break
                    p2 = S[j]
                    found = False
                    for q1 in swap_cand[p1]:
                        if q1 in Sset or q1 == p2:
                            continue
                        for q2 in swap_cand[p2]:
                            if q2 in Sset or q2 == p1 or q2 == q1:
                                continue
                            newS = set(Sset)
                            newS.discard(p1)
                            newS.discard(p2)
                            newS.add(q1)
                            newS.add(q2)
                            ne = _edge_count(newS)
                            if ne > cur:
                                Sset = newS
                                S[i] = q1
                                S[j] = q2
                                cur = ne
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    break
        
        if not improved:
            break
    return Sset, cur

def _simulated_annealing(Sset, pool, swap_cand, deadline, rng):
    """Simulated annealing to escape local optima."""
    S = list(Sset)
    Sset = set(S)
    cur = _edge_count(Sset)
    best = cur
    best_Sset = set(Sset)
    T = 2.0
    cooling = 0.9995
    
    while _now() < deadline - 2.0:
        i = rng.randrange(len(S))
        p = S[i]
        if not swap_cand[p]:
            continue
        q = rng.choice(swap_cand[p])
        if q in Sset:
            continue
        
        newS = set(Sset)
        newS.discard(p)
        newS.add(q)
        ne = _edge_count(newS)
        delta = ne - cur
        
        if delta >= 0 or rng.random() < math.exp(delta / T):
            Sset = newS
            S[i] = q
            cur = ne
            if cur > best:
                best = cur
                best_Sset = set(Sset)
        
        T *= cooling
        if T < 0.005:
            T = 0.005
    
    return best_Sset, best

def construct():
    global _START, _DEADLINE
    if _START is None:
        _START = time.time()
        # Use most of the 150s budget, leaving safety margin
        _DEADLINE = _START + 140.0

    rng = random.Random(42)

    # Candidate pool: hexagon of radius 5 (91 points)
    pool = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 5:
                pool.append((a, b))
    poolset = set(pool)

    # Precompute swap candidates: pool points within hex-distance 3
    swap_cand = {}
    for p in pool:
        swap_cand[p] = [q for q in pool if q != p and _hexdist(p, q) <= 3]

    best = None
    best_edges = -1

    def consider(Sset):
        nonlocal best, best_edges
        if _valid(Sset):
            e = _edge_count(Sset)
            if e > best_edges:
                best_edges = e
                best = [list(t) for t in Sset]

    # Guaranteed optimal: hexagon radius 3 (37 pts) minus 5 boundary points = 32 pts, 76 edges
    # Hex radius 3: all (a,b) with hexdist((0,0),(a,b)) <= 3
    hex3 = [p for p in pool if _hexdist((0,0), p) <= 3]
    # Remove 5 boundary points (hexdist == 3) to get 32 points
    boundary3 = [p for p in hex3 if _hexdist((0,0), p) == 3]
    # Remove specific 5 boundary points to maintain 76 edges
    # Boundary points: (3,0),(3,-1),(3,-2),(3,-3),(2,-3),(1,-3),(0,-3),(-1,-3),(-2,-3),(-3,-3),(-3,-2),(-3,-1),(-3,0),(-3,1),(-3,2),(-3,3),(-2,3),(-1,3),(0,3),(1,3),(2,3),(3,3)
    # Remove 5 that are least connected
    to_remove = [(3,0), (3,-1), (3,-2), (3,-3), (2,-3)]
    hardcoded_optimal = [p for p in hex3 if p not in to_remove]
    
    # Verify hardcoded
    hc_set = set(tuple(p) for p in hardcoded_optimal)
    hc_edges = _edge_count(hc_set)
    if len(hc_set) == 32 and hc_edges >= 76:
        best = hardcoded_optimal
        best_edges = hc_edges

    # If we already have 76, return immediately
    if best_edges >= 76:
        assert len(best) == 32
        assert len(set(tuple(p) for p in best)) == 32
        assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in best)
        return best

    # Search to try to find 76 (in case hardcoded is wrong)
    def grow_to_32(seed):
        Sset = set(seed)
        Sset = {p for p in Sset if p in poolset}
        if len(Sset) >= 32:
            return set(list(Sset)[:32])
        while len(Sset) < 32:
            if _now() > _DEADLINE - 3.0:
                return Sset
            best_pts = []
            best_new = -1
            for q in pool:
                if q in Sset:
                    continue
                new_edges = sum(1 for n in _neighbors(q) if n in Sset)
                if new_edges > best_new:
                    best_new = new_edges
                    best_pts = [q]
                elif new_edges == best_new:
                    best_pts.append(q)
            if not best_pts:
                break
            Sset.add(rng.choice(best_pts))
        return Sset

    # Try several seeds
    center = (0, 0)
    hex2 = [p for p in pool if _hexdist(p, center) <= 2]
    
    seeds = [
        [center],
        list(hex2),
        [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, -1)],
    ]
    for _ in range(3):
        seeds.append(rng.sample(hex2, 5))

    for seed in seeds:
        if _now() > _DEADLINE - 5.0:
            break
        Sset = grow_to_32(seed)
        if len(Sset) != 32:
            continue
        Sset, _ = _fast_local_search(Sset, pool, swap_cand, _DEADLINE, rng)
        consider(Sset)
        if best_edges >= 76:
            break

    # If still below 76, try simulated annealing from best found
    if best_edges < 76 and best is not None:
        if _now() < _DEADLINE - 5.0:
            Sset, _ = _simulated_annealing(set(tuple(p) for p in best), pool, swap_cand, _DEADLINE, rng)
            consider(Sset)

    # Final fallback: use hardcoded if it's better
    if best is None or best_edges < hc_edges:
        best = hardcoded_optimal
        best_edges = hc_edges

    # Final sanity
    if best is None:
        best = [[a, b] for a in range(16) for b in range(2)]
    
    assert len(best) == 32
    assert len(set(tuple(p) for p in best)) == 32
    assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in best)
    return best
