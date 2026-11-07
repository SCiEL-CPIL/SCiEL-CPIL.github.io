import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges to set of tuples (u, v) with u < v
    orig_edge_set = set()
    for u, v in edges:
        if u == v:
            continue
        if u > v:
            u, v = v, u
        orig_edge_set.add((u, v))
    
    best_bicliques = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds
    for seed in range(30):
        random.seed(seed)
        
        # Copy state
        edge_set = set(orig_edge_set)
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        bicliques = []
        
        def count_coverage(L, R):
            """Count edges in L x R that are in edge_set"""
            count = 0
            for u in L:
                for v in R:
                    if u == v:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in edge_set:
                        count += 1
            return count
        
        def try_grow_biclique(L, R, edge_set, adj):
            """Grow biclique by adding vertices that maximize coverage"""
            improved = True
            while improved:
                improved = False
                current_cov = count_coverage(L, R)
                best_gain = 0
                best_v = None
                best_R = None
                
                # Get list of potential v's - those not in L
                pot_vs = [v for v in range(n) if v not in L]
                random.shuffle(pot_vs)
                
                for v in pot_vs:
                    # New R would be intersection of current R with adj[v]
                    R_new = R.intersection(adj[v])
                    if not R_new:
                        continue
                    
                    L_new = L.union([v])
                    new_cov = count_coverage(L_new, R_new)
                    gain = new_cov - current_cov
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_R = R_new
                
                if best_v is not None:
                    L.add(best_v)
                    R = best_R
                    improved = True
            
            return L, R
        
        while edge_set:
            # Find all vertices with remaining edges
            candidates = [u for u in range(n) if adj[u]]
            if not candidates:
                break
            
            # Try multiple starting vertices and pick the best biclique
            best_biclique = None
            best_cov = -1
            
            # Shuffle candidates for diversity
            shuffled_candidates = candidates[:]
            random.shuffle(shuffled_candidates)
            
            # Try top few candidates
            for start_u in shuffled_candidates[:min(10, len(shuffled_candidates))]:
                L = {start_u}
                R = set(adj[start_u])
                
                # Grow the biclique
                L, R = try_grow_biclique(L, R, edge_set, adj)
                
                cov = count_coverage(L, R)
                if cov > best_cov:
                    best_cov = cov
                    best_biclique = (L, R)
            
            if best_biclique is None:
                break
            
            L, R = best_biclique
            bicliques.append([list(L), list(R)])
            
            # Remove covered edges from edge_set and adj
            for u in L:
                for v in R:
                    if u == v:
                        continue
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in edge_set:
                        edge_set.remove((a, b))
                        adj[a].remove(b)
                        adj[b].remove(a)
        
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
            # Early termination if we found a very good solution
            if best_count <= 5:
                break
    
    # Format result
    if best_bicliques is None:
        return []
    
    result = []
    for b in best_bicliques:
        # Ensure lists of ints
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    return result
