"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    m = len(all_edges)
    
    # Helper: check if L x R is a valid biclique (all cross edges exist)
    def is_biclique(L, R):
        for u in L:
            for v in R:
                if v not in adj[u]:
                    return False
        return True
    
    # Helper: get cross edges of biclique [L, R]
    def cross_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                es.add(e)
        return es
    
    # Greedy: repeatedly find the largest biclique covering the most uncovered edges
    def greedy_partition(uncovered, max_iter=500):
        bicliques = []
        unc = set(uncovered)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if not unc:
                break
            
            best_biclique = None
            best_count = 0
            best_edges = set()
            
            # Try all pairs of vertices as potential "core" of a biclique
            # Expand greedily from each pair
            for u in range(n):
                for v in range(n):
                    if u == v:
                        continue
                    if (min(u,v), max(u,v)) not in unc:
                        continue
                    # Start with L={u}, R={v} or L={v}, R={u}
                    for L_init, R_init in [([u], [v]), ([v], [u])]:
                        L = set(L_init)
                        R = set(R_init)
                        # Try to expand L and R
                        changed = True
                        while changed:
                            changed = False
                            # Try adding to L
                            for w in range(n):
                                if w in R or w in L:
                                    continue
                                # w can join L if w is connected to all in R
                                if all(adj[w].x >= 0 and r in adj[w] for r in R):
                                    if all(r in adj[w] for r in R):
                                        L.add(w)
                                        changed = True
                            # Try adding to R
                            for w in range(n):
                                if w in L or w in R:
                                    continue
                                if all(l in adj[w] for l in L):
                                    R.add(w)
                                    changed = True
                        
                        if len(L) == 0 or len(R) == 0:
                            continue
                        
                        # Get cross edges that are uncovered
                        ce = cross_edges(list(L), list(R))
                        ce_unc = ce & unc
                        if len(ce_unc) > best_count:
                            best_count = len(ce_unc)
                            best_biclique = (list(L), list(R))
                            best_edges = ce_unc
            
            if best_biclique is None:
                # Fallback: single edge bicliques
                for e in list(unc):
                    u, v = e
                    bicliques.append([[u], [v]])
                    unc.remove(e)
                break
            
            L, R = best_biclique
            bicliques.append([L, R])
            ce = cross_edges(L, R)
            unc -= ce
        
        return bicliques
    
    # Initial greedy
    best_solution = greedy_partition(set(all_edges))
    best_count = len(best_solution)
    
    # Try multiple random restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Shuffle edge order to get different greedy results
        shuffled = list(all_edges)
        random.shuffle(shuffled)
        
        # Modified greedy: process edges in shuffled order, try to build bicliques
        bicliques = []
        unc = set(all_edges)
        
        for e in shuffled:
            if e not in unc:
                continue
            u, v = e
            # Try to find a biclique containing this edge that covers many uncovered edges
            best_L = None
            best_R = None
            best_c = 0
            
            # Try expanding from (u,v)
            for L_init, R_init in [([u], [v]), ([v], [u])]:
                L = set(L_init)
                R = set(R_init)
                changed = True
                while changed:
                    changed = False
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(r in adj[w] for r in R):
                                L.add(w)
                                changed = True
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(l in adj[w] for l in L):
                                R.add(w)
                                changed = True
                
                ce = cross_edges(list(L), list(R))
                ce_unc = ce & unc
                c = len(ce_unc)
                if c > best_c:
                    best_c = c
                    best_L = list(L)
                    best_R = list(R)
            
            if best_L is not None and best_c > 0:
                bicliques.append([best_L, best_R])
                ce = cross_edges(best_L, best_R)
                unc -= ce
            else:
                bicliques.append([[u], [v]])
                unc.remove(e)
        
        if len(unc) == 0 and len(bicliques) < best_count:
            best_count = len(bicliques)
            best_solution = bicliques
    
    # Local search: try to merge bicliques
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            improved = False
            for i in range(len(best_solution)):
                for j in range(i+1, len(best_solution)):
                    L1, R1 = best_solution[i][0], best_solution[i][1]
                    L2, R2 = best_solution[j][0], best_solution[j][1]
                    
                    # Try merging: new_L = L1+L2, new_R = R1+R2 or other combinations
                    for new_L, new_R in [
                        (list(set(L1)|set(L2)), list(set(R1)|set(R2))),
                        (list(set(L1)|set(R2)), list(set(R1)|set(L2))),
                        (list(set(L1)|set(L2)), list(set(R1)&set(R2))),
                        (list(set(L1)&set(R2)), list(set(R1)&set(L2))),
                    ]:
                        if not new_L or not new_R:
                            continue
                        if is_biclique(new_L, new_R):
                            # Check if this merge is valid: the union of cross edges should equal union of original cross edges
                            ce_orig = cross_edges(L1, R1) | cross_edges(L2, R2)
                            ce_new = cross_edges(new_L, new_R)
                            if ce_new == ce_orig:
                                # Valid merge
                                new_solution = [b for k, b in enumerate(best_solution) if k != i and k != j]
                                new_solution.append([new_L, new_R])
                                best_solution = new_solution
                                best_count = len(best_solution)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    # Verify and return
    result = []
    for L, R in best_solution:
        result.append([[int(x) for x in L], [int(x) for x in R]])
    
    return result
# EVOLVE-BLOCK-END
