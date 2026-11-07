import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix and edge set
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add(_edge_key(u, v))
        
    # Degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    best_bicliques = None
    best_count = float('inf')
    
    # We will run multiple greedy restarts.
    # To improve, we will use a slightly randomized greedy and then try to merge.
    
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        remaining = set(edge_set)
        bicliques = []
        
        while remaining:
            if time.time() > deadline - 3.0:
                break
                
            # Choose a starting edge
            # Heuristic: pick edge with max sum of degrees in remaining graph
            # To add diversity, we can pick from top K
            deg_rem = [0] * n
            for u, v in remaining:
                deg_rem[u] += 1
                deg_rem[v] += 1
            
            # Sort edges by potential
            sorted_edges = sorted(remaining, key=lambda e: deg_rem[e[0]] + deg_rem[e[1]], reverse=True)
            
            # Pick top 5, choose one randomly
            k = min(5, len(sorted_edges))
            if k > 0:
                start_edge = random.choice(sorted_edges[:k])
            else:
                break
                
            u, v = start_edge
            L = {u}
            R = {v}
            
            # Expand to maximal biclique in remaining graph
            changed = True
            while changed:
                if time.time() > deadline - 2.0:
                    break
                changed = False
                
                # Add to L
                new_L = set(L)
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in R in remaining
                    ok = True
                    for r in R:
                        if not adj[w][r] or _edge_key(w, r) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_L.add(w)
                        changed = True
                L = new_L
                
                # Add to R
                new_R = set(R)
                for w in range(n):
                    if w in R or w in L:
                        continue
                    # Check if w is connected to all in L in remaining
                    ok = True
                    for l in L:
                        if not adj[w][l] or _edge_key(w, l) not in remaining:
                            ok = False
                            break
                    if ok:
                        new_R.add(w)
                        changed = True
                R = new_R
            
            # Validate biclique
            valid = True
            covered_edges = set()
            for l in L:
                for r in R:
                    k_e = _edge_key(l, r)
                    if not adj[l][r] or k_e not in remaining:
                        valid = False
                        break
                    covered_edges.add(k_e)
                if not valid:
                    break
            
            if valid and covered_edges:
                bicliques.append((L, R))
                remaining -= covered_edges
            else:
                # Fallback: single edge
                bicliques.append(({u}, {v}))
                remaining.remove(_edge_key(u, v))
                
        # Post-processing: Try to merge bicliques
        # We do this in a loop until no more merges happen
        merged = True
        while merged and time.time() < deadline - 2.0:
            merged = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline - 2.0:
                        break
                    Li, Ri = bicliques[i]
                    Lj, Rj = bicliques[j]
                    
                    # Try merging (Li, Ri) and (Lj, Rj)
                    # New L = Li U Lj, New R = Ri U Rj
                    # Check if this forms a valid biclique in ORIGINAL graph
                    # AND if the set of edges covered is exactly the union of edges from i and j
                    # Since we are partitioning, edges from i and j are disjoint.
                    # We need to ensure no "extra" edges exist in the original graph
                    # between (Li U Lj) and (Ri U Rj) that are not in the union.
                    
                    new_L = Li | Lj
                    new_R = Ri | Rj
                    
                    # Check validity in original graph
                    valid_orig = True
                    all_edges_in_union = set()
                    for l in new_L:
                        for r in new_R:
                            k_e = _edge_key(l, r)
                            if not adj[l][r]:
                                valid_orig = False
                                break
                            all_edges_in_union.add(k_e)
                        if not valid_orig:
                            break
                    
                    if valid_orig:
                        # Check if all_edges_in_union is a subset of the original edges (it is, by adj check)
                        # And if it matches the union of edges covered by biclique i and j
                        # We don't explicitly store the edge sets, but we can reconstruct them
                        edges_i = set()
                        for l in Li:
                            for r in Ri:
                                edges_i.add(_edge_key(l, r))
                        edges_j = set()
                        for l in Lj:
                            for r in Rj:
                                edges_j.add(_edge_key(l, r))
                        
                        if all_edges_in_union == (edges_i | edges_j):
                            # Merge is valid
                            bicliques[i] = (new_L, new_R)
                            bicliques.pop(j)
                            merged = True
                            break # Restart outer loop as indices changed
                if merged:
                    break
                    
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_bicliques = bicliques[:]
            
        if best_count <= 20: # Optimistic lower bound
            break

    if best_bicliques is None:
        # Fallback
        result = []
        for u, v in edges:
            result.append([[u], [v]])
        return result
    
    # Convert to required format
    result = []
    for L, R in best_bicliques:
        result.append([list(L), list(R)])
    
    return result
