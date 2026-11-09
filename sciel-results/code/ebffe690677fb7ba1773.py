import time
import random

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _hexdist(a,b):
    return (abs(a)+abs(b)+abs(a+b))//2

def _count_edges(pts):
    s = set(pts)
    cnt = 0
    for (a,b) in s:
        for da,db in _NEIGH:
            if (a+da, b+db) in s:
                cnt += 1
    return cnt // 2

def construct():
    n = 32
    pts = [(0,0)]
    pts_set = set(pts)
    
    # Greedy construction
    for _ in range(n - 1):
        candidates = []
        for (a,b) in pts:
            for da, db in _NEIGH:
                nb = (a+da, b+db)
                if nb not in pts_set:
                    # Count how many neighbors this candidate has in pts
                    edges = 0
                    for da2, db2 in _NEIGH:
                        if (nb[0]+da2, nb[1]+db2) in pts_set:
                            edges += 1
                    dist = _hexdist(nb[0], nb[1])
                    candidates.append((edges, dist, nb))
        
        if not candidates:
            break
            
        # Remove duplicates and pick best
        # We want max edges, min dist
        # Sort by edges desc, then dist asc
        # Since candidates might have duplicates, we use a dict or set
        best_cand = None
        best_key = None
        seen = set()
        for edges, dist, nb in candidates:
            if nb in seen:
                continue
            seen.add(nb)
            key = (-edges, dist)
            if best_key is None or key < best_key:
                best_key = key
                best_cand = nb
        
        if best_cand:
            pts.append(best_cand)
            pts_set.add(best_cand)
            
    current_edges = _count_edges(pts)
    
    # If we didn't reach 76, do a local search
    if current_edges < 76:
        # Local search: try swapping points
        # Pool of candidates: points near the current set
        pool = set()
        for (a,b) in pts:
            for da, db in _NEIGH:
                pool.add((a+da, b+db))
            # Also add points slightly further to allow reshaping
            for da in range(-2, 3):
                for db in range(-2, 3):
                    pool.add((a+da, b+db))
        
        pool = list(pool)
        rng = random.Random(42)
        deadline = time.time() + 10.0 # Short deadline for polish
        
        best_e = current_edges
        best_pts = pts[:]
        best_set = set(pts)
        
        while time.time() < deadline:
            i = rng.randrange(n)
            cur = best_pts[i]
            cand = pool[rng.randrange(len(pool))]
            
            if cand == cur or cand in best_set:
                continue
                
            # Calculate delta
            # Remove cur, add cand
            # Edges lost: neighbors of cur in best_set
            lost = 0
            for da, db in _NEIGH:
                if (cur[0]+da, cur[1]+db) in best_set:
                    lost += 1
            
            # Edges gained: neighbors of cand in (best_set - {cur})
            gained = 0
            for da, db in _NEIGH:
                nb = (cand[0]+da, cand[1]+db)
                if nb in best_set and nb != cur:
                    gained += 1
            
            delta = gained - lost
            if delta > 0:
                best_e += delta
                best_pts[i] = cand
                best_set.discard(cur)
                best_set.add(cand)
                if best_e == 76:
                    break
                    
    return [[a,b] for (a,b) in best_pts]
