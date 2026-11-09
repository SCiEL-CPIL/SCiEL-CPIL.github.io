"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets for fast operations
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    total_edges = len(edge_set)
    
    # Helper: Get edges of a biclique
    def get_biclique_edges(L, R):
        e_set = set()
        for u in L:
            for v in R:
                if u != v:
                    e_set.add((min(u,v), max(u,v)))
        return e_set

    # Helper: Expand to maximal biclique containing seed (u, v)
    def expand_biclique(u, v):
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L
            new_L_candidates = []
            for w in range(n):
                if w not in L and w not in R:
                    if R.issubset(adj[w]):
                        new_L_candidates.append(w)
            for w in new_L_candidates:
                L.add(w)
                changed = True
            
            # Expand R
            new_R_candidates = []
            for w in range(n):
                if w not in L and w not in R:
                    if L.issubset(adj[w]):
                        new_R_candidates.append(w)
            for w in new_R_candidates:
                R.add(w)
                changed = True
        return L, R

    def solve(deadline):
        remaining_edges = set(edge_set)
        used_edges = set()
        bicliques = []
        
        # Pre-compute degrees for heuristic seed selection
        deg = [len(adj[i]) for i in range(n)]
        
        while remaining_edges:
            if time.time() > deadline:
                break
                
            best_biclique = None
            best_coverage = 0
            best_L = None
            best_R = None
            
            # Sample seeds from remaining edges
            # To ensure progress, we should check all if small, else sample
            seed_list = list(remaining_edges)
            random.shuffle(seed_list)
            
            # Limit number of seeds to check to keep time bounded
            # For n=30, edge count ~150. Checking all is fast.
            for seed in seed_list:
                u, v = seed
                L, R = expand_biclique(u, v)
                E_B = get_biclique_edges(L, R)
                
                # Check conflict with used edges
                if E_B & used_edges:
                    continue
                
                # Calculate coverage of remaining edges
                coverage = len(E_B & remaining_edges)
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_biclique = (L, R)
                    best_L = L
                    best_R = R
            
            if best_biclique is not None:
                L, R = best_biclique
                bicliques.append([sorted(L), sorted(R)])
                E_B = get_biclique_edges(L, R)
                remaining_edges -= E_B
                used_edges |= E_B
            else:
                # Fallback: take a single edge to ensure progress
                # This should rarely happen if graph is connected enough
                seed = next(iter(remaining_edges))
                u, v = seed
                L = {u}
                R = {v}
                bicliques.append([sorted(L), sorted(R)])
                E_B = {(min(u,v), max(u,v))}
                remaining_edges -= E_B
                used_edges |= E_B
                
        return bicliques

    # Run multiple restarts to find better solution
    random.seed(42)
    start_time = time.time()
    deadline = start_time + 50 # Leave 10s margin
    
    best_result = None
    best_count = float('inf')
    
    # Try a few different random seeds for the shuffle in solve
    for i in range(5):
        if time.time() > deadline:
            break
        random.seed(42 + i)
        res = solve(deadline)
        
        # Validate
        if res:
            covered = set()
            valid = True
            for L, R in res:
                if not L or not R:
                    valid = False
                    break
                E_B = get_biclique_edges(L, R)
                if E_B & covered:
                    valid = False
                    break
                if not E_B.issubset(edge_set):
                    valid = False
                    break
                covered |= E_B
            
            if valid and covered == edge_set:
                if len(res) < best_count:
                    best_count = len(res)
                    best_result = res
            else:
                # If invalid, ignore (should not happen with fallback)
                pass
        
    if best_result is None:
        # Emergency fallback: single edges
        best_result = []
        for u, v in edges:
            best_result.append([[u], [v]])
    
    # Convert to required format (integers, lists)
    final_result = []
    for bic in best_result:
        l = [int(x) for x in bic[0]]
        r = [int(x) for x in bic[1]]
        final_result.append([l, r])
        
    return final_result
# EVOLVE-BLOCK-END
