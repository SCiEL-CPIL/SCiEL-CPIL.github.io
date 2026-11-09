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

def _hex_edges(radius):
    """Count edges in a full hexagon of given radius."""
    pts = set()
    for a in range(-radius, radius+1):
        for b in range(-radius, radius+1):
            if _hexdist(a,b) <= radius:
                pts.add((a,b))
    return _count_edges(pts)

def _known_optimal():
    """Return a known 32-point configuration with 76 edges.
    
    Use a hexagon of radius 3 (37 points, 96 edges) and remove 5 boundary points
    that minimize edge loss.
    """
    R = 3
    pts = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                pts.append((a,b))
    s = set(pts)
    full_edges = _count_edges(pts)
    
    # Compute degree of each point
    deg = {}
    for p in pts:
        d = 0
        for da,db in _NEIGH:
            if (p[0]+da, p[1]+db) in s:
                d += 1
        deg[p] = d
    
    # We need to remove 5 points to go from 37 to 32.
    # Edge loss when removing point p = deg(p) - (number of edges to other removed points)
    # For non-adjacent removed points, loss = sum of degrees.
    # Corners have degree 2, edge-midpoints have degree 3, etc.
    
    # Strategy: try removing 5 points with minimum total degree, 
    # preferring non-adjacent ones.
    # Sort by degree ascending, then by hex distance (corners are at max distance)
    pts_sorted = sorted(pts, key=lambda p: (deg[p], -_hexdist(p[0],p[1])))
    
    # Try greedy removal: pick lowest degree point, remove it, recompute
    for attempt in range(200):
        to_remove = set()
        remaining = list(pts)
        rem_set = set(remaining)
        
        for _ in range(5):
            # Find point with min degree in remaining
            best_p = None
            best_d = 999
            for p in remaining:
                d = 0
                for da,db in _NEIGH:
                    if (p[0]+da, p[1]+db) in rem_set:
                        d += 1
                # Prefer removing points that are adjacent to already-removed points
                # (to save edges)
                adj_removed = 0
                for da,db in _NEIGH:
                    if (p[0]+da, p[1]+db) in to_remove:
                        adj_removed += 1
                # Effective loss = d - adj_removed (edges to removed pts are already gone)
                eff_loss = d - adj_removed
                # Add small random perturbation for variety
                if attempt > 0 and random.Random(attempt * 1000 + len(remaining)).random() < 0.3:
                    eff_loss += random.Random(attempt * 1000 + len(remaining)).random()
                if eff_loss < best_d or (eff_loss == best_d and _hexdist(p[0],p[1]) > _hexdist(best_p[0],best_p[1])):
                    best_d = eff_loss
                    best_p = p
            
            if best_p is None:
                break
            to_remove.add(best_p)
            rem_set.discard(best_p)
            remaining.remove(best_p)
        
        result = [p for p in pts if p not in to_remove]
        e = _count_edges(result)
        if e >= 76:
            return result
    
    return None

def _greedy_construct(n=32):
    """Greedy construction: add points one by one, each maximizing new edges."""
    order = _spiral_order(12)
    selected = [order[0]]
    s = set(selected)
    
    for _ in range(1, n):
        best_pt = None
        best_edges = -1
        for pt in order:
            if pt in s:
                continue
            edges = 0
            for da,db in _NEIGH:
                if (pt[0]+da, pt[1]+db) in s:
                    edges += 1
            if edges > best_edges:
                best_edges = edges
                best_pt = pt
        if best_pt is None:
            break
        selected.append(best_pt)
        s.add(best_pt)
    
    return selected

def _local_search_sa(pts, n=32, deadline=None):
    """Simulated annealing local search with O(1) edge-delta evaluation."""
    if deadline is None:
        deadline = time.time() + 115
    
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool: all lattice points within hexdist 8 of any selected point
    pool = set()
    for (a,b) in best:
        for da in range(-8, 9):
            for db in range(-8, 9):
                na, nb = a+da, b+db
                if abs(na) <= 64 and abs(nb) <= 64 and _hexdist(na, nb) <= 8:
                    pool.add((na, nb))
    # Also add a broader set of high-value candidates
    spiral = _spiral_order(10)
    for p in spiral[:100]:
        if abs(p[0]) <= 64 and abs(p[1]) <= 64:
            pool.add(p)
    pool = list(pool)
    pool_idx = {p: i for i, p in enumerate(pool)}
    
    rng = random.Random(42)
    T = 2.0
    T_min = 0.01
    cooling = 0.9995
    iters = 0
    
    while time.time() < deadline:
        iters += 1
        i = rng.randrange(n)
        cur = best[i]
        cand = pool[rng.randrange(len(pool))]
        if cand == cur or cand in in_set:
            continue
        
        # Count edges lost by removing cur
        edges_lost = 0
        for da, db in _NEIGH:
            if (cur[0]+da, cur[1]+db) in in_set:
                edges_lost += 1
        
        # Count edges gained by adding cand
        edges_gained = 0
        for da, db in _NEIGH:
            nb = (cand[0]+da, cand[1]+db)
            if nb in in_set and nb != cur:
                edges_gained += 1
        
        delta = edges_gained - edges_lost
        
        if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
            best_e += delta
            best[i] = cand
            in_set.discard(cur)
            in_set.add(cand)
            if best_e >= 76:
                return best, best_e
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher temperature occasionally
            if iters % 50000 == 0:
                T = 1.0
    
    return best, best_e

def _two_swap_search(pts, n=32, deadline=None):
    """Try removing 2 points and adding 2 different points."""
    if deadline is None:
        deadline = time.time() + 115
    
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Candidate pool
    pool = set()
    for (a,b) in best:
        for da in range(-6, 7):
            for db in range(-6, 7):
                na, nb = a+da, b+db
                if abs(na) <= 64 and abs(nb) <= 64 and _hexdist(na, nb) <= 8:
                    pool.add((na, nb))
    pool = list(pool)
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        cur1 = best[i]
        cur2 = best[j]
        
        cand1 = pool[rng.randrange(len(pool))]
        cand2 = pool[rng.randrange(len(pool))]
        if cand1 == cur1 or cand1 == cur2 or cand2 == cur1 or cand2 == cur2:
            continue
        if cand1 in in_set or cand2 in in_set:
            continue
        if cand1 == cand2:
            continue
        
        # Edges lost
        lost = 0
        for da, db in _NEIGH:
            nb = (cur1[0]+da, cur1[1]+db)
            if nb in in_set and nb != cur2:
                lost += 1
        for da, db in _NEIGH:
            nb = (cur2[0]+da, cur2[1]+db)
            if nb in in_set and nb != cur1:
                lost += 1
        # Edge between cur1 and cur2 (counted twice above, subtract once)
        if (cur1[0]+cur1[1]-cur2[0]-cur2[1], 0) in [(d[0],d[1]) for d in _NEIGH]:
            pass  # check properly
        # Check if cur1 and cur2 are neighbors
        is_adj = False
        for da, db in _NEIGH:
            if (cur1[0]+da, cur1[1]+db) == cur2:
                is_adj = True
                break
        if is_adj:
            lost -= 1  # edge between them was counted in both
        
        # Edges gained
        gained = 0
        temp_set = in_set - {cur1, cur2}
        for da, db in _NEIGH:
            nb = (cand1[0]+da, cand1[1]+db)
            if nb in temp_set:
                gained += 1
        for da, db in _NEIGH:
            nb = (cand2[0]+da, cand2[1]+db)
            if nb in temp_set:
                gained += 1
        # Edge between cand1 and cand2
        for da, db in _NEIGH:
            if (cand1[0]+da, cand1[1]+db) == cand2:
                gained += 1
                break
        
        delta = gained - lost
        if delta > 0:
            best_e += delta
            best[i] = cand1
            best[j] = cand2
            in_set.discard(cur1)
            in_set.discard(cur2)
            in_set.add(cand1)
            in_set.add(cand2)
            if best_e >= 76:
                return best, best_e
    
    return best, best_e

def construct():
    n = 32
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Try known optimal first
    opt = _known_optimal()
    if opt is not None:
        e = _count_edges(opt)
        if e >= 76:
            return [[a,b] for (a,b) in opt]
    
    # Try greedy construction
    greedy = _greedy_construct(n)
    e_greedy = _count_edges(greedy)
    if e_greedy >= 76:
        return [[a,b] for (a,b) in greedy]
    
    # Try spiral order
    order = _spiral_order(10)
    base = order[:n]
    e_base = _count_edges(base)
    if e_base >= 76:
        return [[a,b] for (a,b) in base]
    
    # Try spiral with skips
    for skip in range(1, 8):
        variant = order[skip:skip+n]
        e2 = _count_edges(variant)
        if e2 >= 76:
            return [[a,b] for (a,b) in variant]
    
    # Find best starting point
    best_start = None
    best_start_e = 0
    for candidate in [base, greedy]:
        e_c = _count_edges(candidate)
        if e_c > best_start_e:
            best_start_e = e_c
            best_start = candidate
    
    # Phase 1: Simulated annealing
    result, final_e = _local_search_sa(best_start, n, deadline)
    
    # Phase 2: Two-swap search if not at optimum
    if final_e < 76 and time.time() < deadline - 10:
        result2, final_e2 = _two_swap_search(result, n, deadline)
        if final_e2 > final_e:
            result = result2
            final_e = final_e2
    
    # Phase 3: Another SA round from the improved point
    if final_e < 76 and time.time() < deadline - 5:
        result3, final_e3 = _local_search_sa(result, n, deadline)
        if final_e3 > final_e:
            result = result3
            final_e = final_e3
    
    # Verify and return
    if len(set(result)) != n:
        return [[a,b] for (a,b) in base]
    
    return [[a,b] for (a,b) in result]
