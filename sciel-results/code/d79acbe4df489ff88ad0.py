import time
import random
import numpy as np

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    n = P['n']
    edges = P['edges']
    
    # Build adjacency list as numpy boolean matrix for fast operations
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Seed Solution: One biclique per vertex (star partition)
    seed_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            seed_bicliques.append([[u], right_side])
            
    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Helper: Expand a biclique given remaining edges
    def expand_biclique(start_u, start_v, rem_adj):
        L = {start_u}
        R = {start_v}
        changed = True
        while changed:
            changed = False
            # Expand L
            mask_R = np.zeros(n, dtype=bool)
            for rv in R:
                mask_R[rv] = True
            for u_prime in range(n):
                if u_prime not in L:
                    # Check if u_prime connected to all R in rem_adj
                    if np.all(rem_adj[u_prime, mask_R]):
                        L.add(u_prime)
                        changed = True
            # Expand R
            mask_L = np.zeros(n, dtype=bool)
            for lu in L:
                mask_L[lu] = True
            for v_prime in range(n):
                if v_prime not in R:
                    if np.all(rem_adj[mask_L, v_prime]):
                        R.add(v_prime)
                        changed = True
        return L, R

    # Helper: Remove edges from rem_adj
    def remove_edges(L, R, rem_adj):
        mask_L = np.zeros(n, dtype=bool)
        for u in L:
            mask_L[u] = True
        mask_R = np.zeros(n, dtype=bool)
        for v in R:
            mask_R[v] = True
        rem_adj[mask_L[:, None], mask_R[None, :]] = False
        rem_adj[mask_R[:, None], mask_L[None, :]] = False

    # Helper: Count edges in a biclique
    def biclique_edges(L, R, adj_matrix):
        mask_L = np.zeros(n, dtype=bool)
        for u in L:
            mask_L[u] = True
        mask_R = np.zeros(n, dtype=bool)
        for v in R:
            mask_R[v] = True
        return np.sum(adj_matrix[mask_L[:, None], mask_R[None, :]])

    # Greedy partition
    def greedy_partition(random_restarts=20):
        nonlocal best_bicliques, best_count
        
        for _ in range(random_restarts):
            if time.time() > deadline:
                return
            
            rem_adj = adj.copy()
            current_bicliques = []
            edges_left = len(edges)
            
            while edges_left > 0:
                if time.time() > deadline:
                    return
                
                best_local = None
                best_size = 0
                
                # Find candidate edges to start expansion
                candidates = []
                for u in range(n):
                    for v in range(n):
                        if rem_adj[u, v] and u < v:
                            candidates.append((u, v))
                
                if not candidates:
                    break
                
                random.shuffle(candidates)
                
                for u, v in candidates[:30]:
                    if time.time() > deadline:
                        return
                    if not rem_adj[u, v]:
                        continue
                    
                    L, R = expand_biclique(u, v, rem_adj)
                    size = biclique_edges(L, R, rem_adj)
                    
                    if size > best_size:
                        best_size = size
                        best_local = (L, R)
                
                if not best_local:
                    # Fallback: single edge
                    u, v = candidates[0]
                    best_local = ({u}, {v})
                    best_size = 1
                
                L, R = best_local
                current_bicliques.append([list(L), list(R)])
                remove_edges(L, R, rem_adj)
                edges_left -= best_size
            
            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    return

    # Local Search: Try to merge bicliques
    def try_merge(bicliques, rem_adj):
        nonlocal best_bicliques, best_count
        if time.time() > deadline:
            return
        n_b = len(bicliques)
        if n_b < 2:
            return
        
        for i in range(n_b):
            for j in range(i+1, n_b):
                if time.time() > deadline:
                    return
                
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Try to merge: L1+L2, R1+R2
                # Check if all cross edges exist in original graph
                L_new = set(L1) | set(L2)
                R_new = set(R1) | set(R2)
                
                # Check if (L_new, R_new) is a valid biclique in original graph
                # And if the edges covered by new biclique are subset of edges covered by old bicliques
                # Actually, we want the new biclique to cover exactly the union of edges from i and j
                
                # Let's check if all pairs in L_new x R_new are edges in the original graph
                valid = True
                for u in L_new:
                    for v in R_new:
                        if not adj[u, v]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check if the edges covered by (L_new, R_new) are exactly the union of edges from bicliques i and j
                # We need to be careful: the edges in (L_new, R_new) must be a subset of the edges originally covered by bicliques i and j
                # Let's compute the edges covered by bicliques i and j
                edges_ij = set()
                for u in L1:
                    for v in R1:
                        edges_ij.add((u, v))
                        edges_ij.add((v, u))
                for u in L2:
                    for v in R2:
                        edges_ij.add((u, v))
                        edges_ij.add((v, u))
                
                edges_new = set()
                for u in L_new:
                    for v in R_new:
                        edges_new.add((u, v))
                        edges_new.add((v, u))
                
                if edges_new == edges_ij:
                    # Valid merge
                    new_bicliques = []
                    for k in range(n_b):
                        if k != i and k != j:
                            new_bicliques.append(bicliques[k])
                    new_bicliques.append([list(L_new), list(R_new)])
                    
                    if len(new_bicliques) < best_count:
                        best_count = len(new_bicliques)
                        best_bicliques = new_bicliques
                        if best_count == 1:
                            return
                    return  # Only do one merge per call

    # Main
    greedy_partition(random_restarts=30)
    
    # Try local search improvements
    for _ in range(10):
        if time.time() > deadline:
            break
        if best_count < 2:
            break
        rem_adj = adj.copy()
        # We need to track which edges are covered by which biclique to check merge validity properly
        # For now, just try merging on the current best solution
        try_merge(best_bicliques, rem_adj)
    
    # Return best found
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
