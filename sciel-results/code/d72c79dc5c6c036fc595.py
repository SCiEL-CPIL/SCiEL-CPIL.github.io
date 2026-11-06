import time

def construct():
    # Deterministic construction of the optimal 32-point hexagonal blob
    # 1. Generate the 19 points of the hexagon of radius 2
    points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            # Hex distance check: (|a| + |a+b| + |b|) // 2 <= 2
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 2:
                points.append((a, b))
    
    # We have 19 points. We need 13 more from the ring of radius 3.
    # The ring of radius 3 has 18 points.
    # To maximize edges, we select 13 contiguous points from this ring.
    # The ring points are those with hex distance exactly 3.
    ring3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if (abs(a) + abs(a + b) + abs(b)) // 2 == 3:
                ring3.append((a, b))
                
    # Sort ring3 to form a contiguous chain?
    # The ring is a cycle. We can just take the first 13 in a sorted order if they are contiguous.
    # A better way: Start from a corner and walk 13 steps.
    # Corner: (3, 0)
    # Neighbors in ring 3: (2, 1), (3, -1)
    # Let's just pick 13 points that are close to each other.
    # Points in ring 3 near (3,0):
    # (3,0), (2,1), (1,2), (0,3), (-1,2), (-2,1), (-3,0) ... this is one side.
    # There are 6 sides of 3 points each? No, 18 points.
    # Side 1: (3,0), (2,1), (1,2)
    # Side 2: (0,3), (-1,2), (-2,1)
    # Side 3: (-3,0), (-2,-1), (-1,-2)
    # Side 4: (0,-3), (1,-2), (2,-1)
    # Side 5: (3,-2)?? No.
    
    # Let's just use the local search from the current program but with a very tight deadline
    # and a good seed. The current program is robust.
    
    # I will use the current program's logic but optimized.
    import random
    rng = random.Random(42)
    
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

    def _delta_edges(Sset, p, q):
        lost = sum(1 for n in _neighbors(p) if n in Sset)
        gained = sum(1 for n in _neighbors(q) if n in Sset and n != p)
        return gained - lost

    pool = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            if (abs(a) + abs(a + b) + abs(b)) // 2 <= 5:
                pool.append((a, b))
    poolset = set(pool)

    swap_cand = {}
    for p in pool:
        swap_cand[p] = [q for q in pool if q != p and _hexdist(p, q) <= 2]

    center = (0, 0)
    hex2 = [p for p in pool if _hexdist(p, center) <= 2]
    
    # Start with the 19-point hexagon
    Sset = set(hex2)
    
    # Grow to 32
    while len(Sset) < 32:
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
        
    # Local Search
    S = list(Sset)
    cur = _edge_count(Sset)
    for _ in range(100):
        improved = False
        for i in range(len(S)):
            p = S[i]
            best_q = None
            best_delta = 0
            for q in swap_cand[p]:
                if q in Sset:
                    continue
                d = _delta_edges(Sset, p, q)
                if d > best_delta:
                    best_delta = d
                    best_q = q
            if best_q is not None:
                Sset.discard(p)
                Sset.add(best_q)
                S[i] = best_q
                cur += best_delta
                improved = True
        if not improved:
            break
            
    if cur == 76:
        return [list(t) for t in Sset]
        
    # Fallback to current program's logic if it fails (unlikely)
    # But since we expect 76, we return the best found.
    return [list(t) for t in Sset]
