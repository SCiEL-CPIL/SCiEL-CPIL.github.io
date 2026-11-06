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
    # Hardcoded optimal solution: 32 points from a hexagon of radius 3
    # with 5 specific boundary points removed to maximize edges (76).
    # Hex radius 3 points:
    hex3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 3:
                hex3.append((a, b))
    
    # Remove 5 boundary points to get 32 points
    # These are chosen to minimize edge loss
    to_remove = {(3,0), (3,-1), (3,-2), (3,-3), (2,-3)}
    hardcoded_optimal = [p for p in hex3 if p not in to_remove]
    
    # Convert to list of lists as required
    result = [[a, b] for a, b in hardcoded_optimal]
    
    # Verify constraints
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    assert all(abs(p[0]) <= 64 and abs(p[1]) <= 64 for p in result)
    
    return result
