"""Biclique partition via Greedy + Local Search Merging."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def _can_merge(b1, b2, adj_matrix):
    """Check if two bicliques can be merged."""
    L1, R1 = b1
    L2, R2 = b2
    
    L_new = set(L1) | set(L2)
    R_new = set(R1) | set(R2)
    
    for u in L_new:
        for v in R_new:
            if not adj_matrix[u][v]:
                return False
    return True

def _greedy_construct(adj_matrix, n, edges, random_seed=None):
    """Construct initial partition using greedy maximal biclique selection."""
    if random_seed is not None:
        random.seed(random_seed)
        
    remaining = set((min(u,v), max(u,v)) for u,v in edges)
    bicliques = []
    
    rem_adj = [set() for _ in range(n)]
    for u, v in remaining:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    while remaining:
        best_e = None
        best_score = -1
        for e in remaining:
            u, v = e
            score = len(rem_adj[u]) * len(rem_adj[v])
            if score > best_score:
                best_score = score
                best_e = e
                
        if best_e is None:
            break
            
        u0, v0 = best_e
        L = {u0}
        R = {v0}
        
        changed = True
        while changed:
            changed = False
            best_l_gain = 0
            best_l_v = None
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(r in rem_adj[w] for r in R):
                    gain = len(R)
                    if gain > best_l_gain:
                        best_l_gain = gain
                        best_l_v = w
            if best_l_v is not None:
                L.add(best_l_v)
                changed = True
                continue
                
            best_r_gain = 0
            best_r_v = None
            for w in range(n):
                if w in L or w in R:
                    continue
                if all(l in rem_adj[w] for l in L):
                    gain = len(L)
                    if gain > best_r_gain:
                        best_r_gain = gain
                        best_r_v = w
            if best_r_v is not None:
                R.add(best_r_v)
                changed = True
                
        if L and R:
            bicliques.append([sorted(L), sorted(R)])
            for u in L:
                for v in R:
                    e = (min(u,v), max(u,v))
                    if e in remaining:
                        remaining.discard(e)
                        rem_adj[u].discard(v)
                        rem_adj[v].discard(u)
                        
    for e in remaining:
        u, v = e
        bicliques.append([[u], [v]])
        
    return bicliques

def _local_search(bicliques, adj_matrix, random_seed=None):
    """Try to merge bicliques to reduce count with randomization."""
    if random_seed is not None:
        random.seed(random_seed)
        
    bicliques = [b[:] for b in bicliques]
    changed = True
    while changed:
        changed = False
        pairs = [(i, j) for i in range(len(bicliques)) for j in range(i + 1, len(bicliques))]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if _can_merge(bicliques[i], bicliques[j], adj_matrix):
                L_new = sorted(set(bicliques[i][0]) | set(bicliques[j][0]))
                R_new = sorted(set(bicliques[i][1]) | set(bicliques[j][1]))
                bicliques[i] = [L_new, R_new]
                bicliques.pop(j)
                changed = True
                break
    return bicliques

def construct():
    n = P['n']
    edges = P['edges']
    
    adj_matrix = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True
        
    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 55.0
    
    seed = 42
    while time.time() - start_time < timeout:
        bicliques = _greedy_construct(adj_matrix, n, edges, random_seed=seed)
        bicliques = _local_search(bicliques, adj_matrix, random_seed=seed)
        
        count = len(bicliques)
        if count < best_count:
            best_count = count
            best_bicliques = bicliques
            
        seed += 1
        if best_count == 1:
            break
                 
    if best_bicliques is None:
        result = []
        for u, v in edges:
            result.append([[int(u)], [int(v)]])
        return result
    
    result = []
    for L, R in best_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        result.append([L_int, R_int])
        
    return result
# EVOLVE-BLOCK-END
