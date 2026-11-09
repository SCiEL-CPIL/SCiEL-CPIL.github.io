import time
import random
import numpy as np
from itertools import combinations, permutations

Q3_EDGES = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
Q3_A = [0, 3, 5, 6]
Q3_B = [1, 2, 4, 7]
q3_adj = np.zeros((8,8), dtype=bool)
for u, v in Q3_EDGES:
    q3_adj[u, v] = True
    q3_adj[v, u] = True

SUBSETS = list(combinations(range(16), 8))

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_edges = 39
    best_graph = [[a, b] for a in range(3) for b in range(3, 16)]
    
    def contains_q3_fast(adj, subset):
        # Degree check
        for u in subset:
            d = 0
            for v in subset:
                if u != v:
                    d += adj[u, v]
            if d < 3:
                return False
                
        # Try bipartite mappings
        verts = list(range(8))
        for a_idx in combinations(verts, 4):
            b_idx = tuple(v for v in verts if v not in a_idx)
            
            for pa in permutations(a_idx):
                for pb in permutations(b_idx):
                    valid = True
                    for i in range(4):
                        u_q3 = Q3_A[i]
                        u_sub = subset[pa[i]]
                        for j in range(4):
                            v_q3 = Q3_B[j]
                            v_sub = subset[pb[j]]
                            if q3_adj[u_q3, v_q3]:
                                if adj[u_sub, v_sub] == 0:
                                    valid = False
                                    break
                        if not valid: break
                    if valid:
                        return True
        return False

    while time.time() - start_time < time_limit:
        adj = np.ones((16, 16), dtype=int)
        np.fill_diagonal(adj, 0)
        edge_count = 120
        
        found = True
        while found and time.time() - start_time < time_limit:
            found = False
            idxs = list(range(len(SUBSETS)))
            random.shuffle(idxs)
            
            for i in idxs:
                subset = SUBSETS[i]
                
                # Quick degree check
                skip = False
                for u in subset:
                    d = 0
                    for v in subset:
                        if u != v:
                            d += adj[u, v]
                    if d < 3:
                        skip = True
                        break
                if skip:
                    continue
                    
                # Edge count check
                ec = 0
                for u in subset:
                    for v in subset:
                        if u < v:
                            ec += adj[u, v]
                if ec < 12:
                    continue
                    
                if contains_q3_fast(adj, subset):
                    edges_in_subset = []
                    for u in subset:
                        for v in subset:
                            if u < v and adj[u, v] == 1:
                                edges_in_subset.append((u, v))
                    if edges_in_subset:
                        u, v = random.choice(edges_in_subset)
                        adj[u, v] = 0
                        adj[v, u] = 0
                        edge_count -= 1
                        found = True
                        break
            
        if edge_count > best_edges:
            best_edges = edge_count
            best_graph = []
            for u in range(16):
                for v in range(u+1, 16):
                    if adj[u, v] == 1:
                        best_graph.append([u, v])

    return best_graph
