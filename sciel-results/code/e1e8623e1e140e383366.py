import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _expand_biclique(n, adj, start_u, start_v, deadline):
    """Expand a seed edge to a maximal biclique."""
    L = {start_u}
    R = {start_v}
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try adding to L
        for w in range(n):
            if w in L or w in R:
                continue
            # w can join L if connected to all in R
            if all(adj[w][x] for x in R):
                L.add(w)
                changed = True
        # Try adding to R
        for w in range(n):
            if w in L or w in R:
                continue
            # w can join R if connected to all in L
            if all(adj[w][x] for x in L):
                R.add(w)
                changed = True
    return L, R

def _generate_candidates(n, edges, adj, num_samples, deadline):
    """Generate a diverse set of candidate bicliques."""
    candidates = []
    seen = set()
    
    # Always include some "star" bicliques or small ones? 
    # Actually, maximal bicliques from random edges should cover the space.
    
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        # Pick a random edge
        u, v = random.choice(edges)
        L, R = _expand_biclique(n, adj, u, v, deadline)
        
        # Canonical form
        L_sorted = tuple(sorted(L))
        R_sorted = tuple(sorted(R))
        if L_sorted[0] > R_sorted[0]:
            L_sorted, R_sorted = R_sorted, L_sorted
        
        key = (L_sorted, R_sorted)
        if key not in seen:
            seen.add(key)
            candidates.append((set(L_sorted), set(R_sorted)))
            
    return candidates

def _greedy_set_cover(candidates, edge_set, deadline):
    """Greedy set cover on candidates."""
    remaining = set(edge_set)
    chosen = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_biclique = None
        best_count = 0
        best_size = 0 # Tie breaker: prefer larger bicliques (more vertices)
        
        for L, R in candidates:
            # Count how many edges from this biclique are still uncovered
            count = 0
            for u in L:
                for v in R:
                    k = _edge_key(u, v)
                    if k in remaining:
                        count += 1
            
            if count > best_count or (count == best_count and len(L) + len(R) > best_size):
                best_count = count
                best_size = len(L) + len(R)
                best_biclique = (L, R)
        
        if best_biclique is None:
            break
        
        L, R = best_biclique
        chosen.append((L, R))
        
        # Remove covered edges
        for u in L:
            for v in R:
                k = _edge_key(u, v)
                remaining.discard(k)
    
    return chosen

def _try_merge(partition, adj, deadline):
    """Try to merge pairs of bicliques."""
    merged = True
    current = [list(b) for b in partition]
    
    while merged and time.time() < deadline:
        merged = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                
                Li, Ri = current[i]
                Lj, Rj = current[j]
                
                # Try 4 combinations
                combos = [
                    (set(Li) | set(Lj), set(Ri) | set(Rj)),
                    (set(Li) | set(Rj), set(Ri) | set(Lj)),
                    (set(Ri) | set(Lj), set(Li) | set(Rj)),
                    (set(Ri) | set(Rj), set(Li) | set(Lj))
                ]
                
                for NL, NR in combos:
                    if not NL or not NR:
                        continue
                    if NL & NR:
                        continue
                    
                    # Check if valid biclique
                    valid = True
                    for u in NL:
                        for v in NR:
                            if not adj[u][v]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if not valid:
                        continue
                    
                    # Check edge coverage matches
                    cross_new = set()
                    for u in NL:
                        for v in NR:
                            cross_new.add(_edge_key(u, v))
                    
                    cross_i = set()
                    for u in Li:
                        for v in Ri:
                            cross_i.add(_edge_key(u, v))
                    cross_j = set()
                    for u in Lj:
                        for v in Rj:
                            cross_j.add(_edge_key(u, v))
                    
                    if cross_new == (cross_i | cross_j):
                        current[i] = [list(NL), list(NR)]
                        current.pop(j)
                        merged = True
                        break
            if merged:
                break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add(_edge_key(u, v))
    
    best_result = None
    best_len = float('inf')
    
    # Multiple random trials
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        # Generate candidates
        # Increase samples in later trials? Or keep constant.
        samples = 50 + trial * 5
        candidates = _generate_candidates(n, edges, adj, samples, deadline - 10.0)
        
        # Greedy cover
        result = _greedy_set_cover(candidates, edge_set, deadline - 5.0)
        
        # Merge
        result = _try_merge(result, adj, deadline - 2.0)
        
        # Validate
        covered = set()
        valid = True
        for L, R in result:
            if not L or not R:
                valid = False
                break
            if set(L) & set(R):
                valid = False
                break
            for u in L:
                for v in R:
                    k = _edge_key(u, v)
                    if k not in edge_set or k in covered:
                        valid = False
                        break
                    covered.add(k)
            if not valid:
                break
        
        if valid and covered == edge_set:
            if len(result) < best_len:
                best_len = len(result)
                best_result = result
                
    if best_result is None:
        # Fallback: individual edges
        best_result = [[[u], [v]] for u, v in edges]
        
    # Ensure Python int types
    final = []
    for L, R in best_result:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        final.append([L_int, R_int])
        
    return final
