"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

import random
import time
from itertools import combinations

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set((min(a,b), max(a,b)) for a,b in edges)
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        
    def is_biclique(L, R, edge_set):
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_set:
                    return False
        return True

    def greedy_cover(uncovered_edges):
        """Greedy cover of uncovered_edges. Returns list of [L, R]."""
        if not uncovered_edges:
            return []
        uncovered = set(uncovered_edges)
        bicliques = []
        
        while uncovered:
            best_biclique = None
            best_size = 0
            
            # Sample a subset of edges to speed up if needed, but n=30 is small
            # We iterate over all edges in uncovered to find the best seed
            for e in list(uncovered):
                u, v = e
                L = {u}
                R = {v}
                changed = True
                while changed:
                    changed = False
                    # Add to L
                    for w in range(n):
                        if w not in L and w not in R:
                            if all((min(w,x), max(w,x)) in uncovered for x in R):
                                L.add(w)
                                changed = True
                    # Add to R
                    for w in range(n):
                        if w not in L and w not in R:
                            if all((min(w,x), max(w,x)) in uncovered for x in L):
                                R.add(w)
                                changed = True
                
                # Verify
                valid = True
                for a in L:
                    for b in R:
                        if (min(a,b), max(a,b)) not in uncovered:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_biclique = (L, R)
            
            if best_biclique is None:
                break
            
            L, R = best_biclique
            bicliques.append([list(L), list(R)])
            
            # Remove covered edges
            for a in L:
                for b in R:
                    uncovered.discard((min(a,b), max(a,b)))
        
        return bicliques

    def optimize(bicliques):
        """Remove redundant bicliques and merge where possible."""
        # Convert to set of edges covered by each
        biclique_edges = []
        for L, R in bicliques:
            es = set()
            for u in L:
                for v in R:
                    es.add((min(u,v), max(u,v)))
            biclique_edges.append(es)
        
        # Remove redundant
        new_bicliques = []
        new_biclique_edges = []
        for i in range(len(bicliques)):
            redundant = False
            for j in range(len(bicliques)):
                if i == j:
                    continue
                if biclique_edges[i].issubset(biclique_edges[j]):
                    redundant = True
                    break
            if not redundant:
                new_bicliques.append(bicliques[i])
                new_biclique_edges.append(biclique_edges[i])
        
        # Merge
        changed = True
        while changed:
            changed = False
            for i in range(len(new_bicliques)):
                for j in range(i+1, len(new_bicliques)):
                    L1, R1 = new_bicliques[i]
                    L2, R2 = new_bicliques[j]
                    L_union = set(L1) | set(L2)
                    R_union = set(R1) | set(R2)
                    
                    # Check if union is a valid biclique in the ORIGINAL graph
                    # Note: We must ensure all edges in L_union x R_union are in the graph
                    valid = True
                    for u in L_union:
                        for v in R_union:
                            if (min(u,v), max(u,v)) not in edge_set:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        # Merge
                        new_bicliques[i] = [list(L_union), list(R_union)]
                        new_biclique_edges[i] = new_biclique_edges[i] | new_biclique_edges[j]
                        
                        # Remove j
                        new_bicliques.pop(j)
                        new_biclique_edges.pop(j)
                        changed = True
                        break
                if changed:
                    break
        
        return new_bicliques

    # Initial Solution
    start_time = time.time()
    best_sol = greedy_cover(edge_set)
    best_sol = optimize(best_sol)
    best_len = len(best_sol)
    
    # Local Search
    while time.time() - start_time < 50:
        # Perturb: Remove a random biclique and re-cover its edges
        if not best_sol:
            break
            
        idx = random.randint(0, len(best_sol) - 1)
        removed = best_sol[idx]
        
        # Find edges covered by removed
        rem_edges = set()
        for u in removed[0]:
            for v in removed[1]:
                rem_edges.add((min(u,v), max(u,v)))
                
        # Re-cover these edges
        new_cover = greedy_cover(rem_edges)
        
        # Combine
        candidate = best_sol[:idx] + best_sol[idx+1:] + new_cover
        candidate = optimize(candidate)
        
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_sol = candidate
            
    # Final Format
    result = []
    for L, R in best_sol:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])
    
    return result

# EVOLVE-BLOCK-END
