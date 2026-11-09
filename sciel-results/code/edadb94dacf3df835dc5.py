import math
import time
import itertools

_NEIGH = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

def _dist_sq(a1, b1, a2, b2):
    da, db = a1 - a2, b1 - b2
    return da * da + da * db + db * db

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
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n = 32
    
    # Strategy 1: Exact search over Hexagon of Radius 3 (37 points)
    # We need to choose 32 points, which is equivalent to removing 5 points.
    # C(37, 5) = 435,897 combinations. This is fast.
    
    R1 = 3
    hex1 = []
    for a in range(-R1, R1+1):
        for b in range(-R1, R1+1):
            if _hexdist(a,b) <= R1:
                hex1.append((a,b))
    
    if len(hex1) != 37:
        # Fallback if hexagon size is unexpected
        pass
    else:
        hex_set = set(hex1)
        # Precompute adjacency and degrees within hex1
        deg = {}
        adj = {}
        for p in hex1:
            neighbors = []
            for da, db in _NEIGH:
                np_ = (p[0]+da, p[1]+db)
                if np_ in hex_set:
                    neighbors.append(np_)
            deg[p] = len(neighbors)
            adj[p] = set(neighbors)
        
        # Total edges in full hexagon
        E_full = sum(deg.values()) // 2
        
        best_e = -1
        best_keep = None
        
        # Iterate over all combinations of 5 points to remove
        # Using indices for speed
        indices = range(37)
        
        # Optimization: Precompute point indices for the remove set
        # We want to maximize E_kept = E_full - Sum(deg[p] for p in R) + E(R, R)
        # where R is the set of removed points.
        
        for remove_idx in itertools.combinations(indices, 5):
            sum_deg = 0
            remove_pts = []
            for idx in remove_idx:
                p = hex1[idx]
                sum_deg += deg[p]
                remove_pts.append(p)
            
            # Calculate edges within the removed set E(R, R)
            e_rr = 0
            for i in range(5):
                p_i = remove_pts[i]
                adj_i = adj[p_i]
                for j in range(i+1, 5):
                    p_j = remove_pts[j]
                    if p_j in adj_i:
                        e_rr += 1
                        
            e_kept = E_full - sum_deg + e_rr
            
            if e_kept > best_e:
                best_e = e_kept
                # Store the indices to keep
                remove_set = set(remove_idx)
                keep_indices = [i for i in indices if i not in remove_set]
                best_keep = [hex1[i] for i in keep_indices]
                
                # Early exit if we hit the theoretical maximum
                if best_e == 76:
                    break
                    
        if best_keep is not None and best_e >= 70: # Threshold for accepting this solution
            # Verify validity just in case
            if len(best_keep) == n:
                # Check distinct and bounds
                pts_set = set(best_keep)
                if len(pts_set) == n:
                    # Check distance constraint (should be satisfied by construction of hexagon? 
                    # No, hexagon points are distinct, but we must ensure no two are distance < 1.
                    # In triangular lattice, min distance is 1. So any distinct points are >= 1 apart.
                    # Wait, the constraint is "mutual distance at least one". 
                    # Lattice points a+b*omega have integer coordinates.
                    # Distance squared is da^2 + da*db + db^2.
                    # Min non-zero distance squared is 1 (e.g., (1,0)).
                    # So any distinct lattice points are at distance >= 1.
                    # Thus, any subset of lattice points is valid.
                    return [[a,b] for (a,b) in best_keep]

    # Strategy 2: Fallback to Local Search on larger pool if Strategy 1 failed or score was low
    # This is unlikely to be needed if Strategy 1 works, but provides robustness.
    
    # Build a good initial configuration using spiral order
    R = 4
    cands = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if _hexdist(a,b) <= R:
                hd = _hexdist(a,b)
                x = a + 0.5*b
                y = 0.8660254037844386*b
                ang = math.atan2(y, x)
                cands.append((hd, ang, a, b))
                
    cands.sort()
    best = [(a,b) for hd, ang, a, b in cands[:n]]
    best_set = set(best)
    current_edges = _count_edges(best)
    
    # Candidate pool for swaps
    R2 = 6
    all_candidates = []
    for a in range(-R2, R2+1):
        for b in range(-R2, R2+1):
            if _hexdist(a,b) <= R2:
                all_candidates.append((a, b))
    
    # Local Improvement
    improved = True
    iteration = 0
    max_iterations = 20000
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        for i in range(len(best)):
            if time.time() > deadline:
                break
            old_pt = best[i]
            
            for new_pt in all_candidates:
                if new_pt == old_pt or new_pt in best_set:
                    continue
                
                # Check minimum distance constraint: dist >= 1 from all other placed points
                valid = True
                for j in range(len(best)):
                    if i == j:
                        continue
                    if _dist_sq(new_pt[0], new_pt[1], best[j][0], best[j][1]) < 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Calculate edge change
                edges_lost = 0
                for da, db in _NEIGH:
                    if (old_pt[0]+da, old_pt[1]+db) in best_set:
                        edges_lost += 1
                        
                edges_gained = 0
                for da, db in _NEIGH:
                    nb = (new_pt[0]+da, new_pt[1]+db)
                    if nb in best_set and nb != old_pt:
                        edges_gained += 1
                        
                delta = edges_gained - edges_lost
                if delta > 0:
                    best[i] = new_pt
                    best_set.remove(old_pt)
                    best_set.add(new_pt)
                    current_edges += delta
                    improved = True
                    break
            if improved:
                break

    # Final verification
    if len(best) != n:
        best = [(a,b) for hd, ang, a, b in cands[:n]]
        
    return [[a,b] for (a,b) in best]
