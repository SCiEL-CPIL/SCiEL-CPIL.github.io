import numpy as np
import time
import itertools

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    # Precompute triangle indices
    tri_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                tri_list.append((i, j, k))
    tri_arr = np.array(tri_list)
    I = tri_arr[:, 0]
    J = tri_arr[:, 1]
    K = tri_arr[:, 2]
    n_tri = len(tri_arr)
    
    A = list(range(8))
    B = list(range(8, 16))
    all_cross = [(u, v) for u in A for v in B]
    all_int_A = list(itertools.combinations(A, 2))
    all_int_B = list(itertools.combinations(B, 2))
    
    def get_score(adj):
        deg = adj.sum(axis=1)
        valid = (adj[I, J] == 1) & (adj[J, K] == 1) & (adj[K, I] == 1)
        if not np.any(valid):
            return 0
        sums = deg[I[valid]] + deg[J[valid]] + deg[K[valid]]
        return int(np.max(sums))
    
    def build_adj(removed_cross, int_A, int_B):
        adj = np.zeros((n, n), dtype=np.int8)
        for u in range(8):
            for v in range(8, 16):
                adj[u, v] = 1
                adj[v, u] = 1
        for u, v in removed_cross:
            adj[u, v] = 0
            adj[v, u] = 0
        for u, v in int_A:
            adj[u, v] = 1
            adj[v, u] = 1
        for u, v in int_B:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    # Phase 1: Systematic search over small perturbations of K_{8,8}
    # Try removing k cross edges and adding k+1 internal edges
    best_adj = None
    best_score = 999
    
    # k=0: baseline (64 cross + 1 internal = 65)
    for u, v in all_int_A:
        adj = build_adj([], [(u, v)], [])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    for u, v in all_int_B:
        adj = build_adj([], [], [(u, v)])
        if adj.sum() // 2 == target_edges:
            s = get_score(adj)
            if s < best_score:
                best_score = s
                best_adj = adj.copy()
    
    # k=1: remove 1 cross, add 2 internal
    if time.time() - start_time < deadline - 60:
        for rc in all_cross:
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj([rc], [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            # Also try 2 in A
            for ia1, ia2 in itertools.combinations(all_int_A, 2):
                adj = build_adj([rc], [ia1, ia2], [])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            # Also try 2 in B
            for ib1, ib2 in itertools.combinations(all_int_B, 2):
                adj = build_adj([rc], [], [ib1, ib2])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            if time.time() - start_time > deadline - 60:
                break
    
    # k=2: remove 2 cross, add 3 internal
    if time.time() - start_time < deadline - 40:
        for rc1, rc2 in itertools.combinations(all_cross, 2):
            removed = [rc1, rc2]
            # Try various internal combinations
            for ia in all_int_A:
                for ib in all_int_B:
                    adj = build_adj(removed, [ia], [ib])
                    if adj.sum() // 2 == target_edges:
                        s = get_score(adj)
                        if s < best_score:
                            best_score = s
                            best_adj = adj.copy()
            for ia1, ia2 in itertools.combinations(all_int_A, 2):
                adj = build_adj(removed, [ia1, ia2], [])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            for ib1, ib2 in itertools.combinations(all_int_B, 2):
                adj = build_adj(removed, [], [ib1, ib2])
                if adj.sum() // 2 == target_edges:
                    s = get_score(adj)
                    if s < best_score:
                        best_score = s
                        best_adj = adj.copy()
            if time.time() - start_time > deadline - 40:
                break
    
    # Phase 2: Simulated annealing with structured moves
    if best_adj is None:
        best_adj = build_adj([], [(0, 1)], [])
        best_score = get_score(best_adj)
    
    current_adj = best_adj.copy()
    current_score = best_score
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.3
    
    # Precompute edge lists for faster access
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 0:
                    non_edges.append((i, j))
        return non_edges
    
    # Structured moves: swap a cross edge for an internal edge or vice versa
    while time.time() - start_time < deadline - 20:
        # Choose move type
        r = np.random.random()
        
        if r < 0.4:
            # Remove a cross edge, add an internal edge
            cross_edges = [(u, v) for u, v in all_cross if current_adj[u, v] == 1]
            if not cross_edges:
                continue
            u, v = cross_edges[np.random.randint(len(cross_edges))]
            
            int_options = [(a, b) for a, b in all_int_A if current_adj[a, b] == 0] + \
                         [(a, b) for a, b in all_int_B if current_adj[a, b] == 0]
            if not int_options:
                continue
            x, y = int_options[np.random.randint(len(int_options))]
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
                    
        elif r < 0.7:
            # Remove an internal edge, add a cross edge
            int_edges = [(a, b) for a, b in all_int_A if current_adj[a, b] == 1] + \
                        [(a, b) for a, b in all_int_B if current_adj[a, b] == 1]
            if not int_edges:
                continue
            x, y = int_edges[np.random.randint(len(int_edges))]
            
            cross_options = [(u, v) for u, v in all_cross if current_adj[u, v] == 0]
            if not cross_options:
                continue
            u, v = cross_options[np.random.randint(len(cross_options))]
            
            current_adj[x, y] = 0
            current_adj[y, x] = 0
            current_adj[u, v] = 1
            current_adj[v, u] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[x, y] = 1
                    current_adj[y, x] = 1
                    current_adj[u, v] = 0
                    current_adj[v, u] = 0
                    
        else:
            # General edge swap
            edges = get_edges(current_adj)
            if not edges:
                continue
            u, v = edges[np.random.randint(len(edges))]
            
            non_edges = get_non_edges(current_adj)
            if not non_edges:
                continue
            x, y = non_edges[np.random.randint(len(non_edges))]
            
            current_adj[u, v] = 0
            current_adj[v, u] = 0
            current_adj[x, y] = 1
            current_adj[y, x] = 1
            
            new_score = get_score(current_adj)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj.copy()
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                    current_adj[x, y] = 0
                    current_adj[y, x] = 0
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Phase 3: Exhaustive local search on best_adj
    if time.time() - start_time < deadline - 5:
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            
            for u, v in edges:
                if time.time() - start_time > deadline - 5:
                    break
                for x, y in non_edges:
                    if x == u or x == v or y == u or y == v:
                        continue
                    
                    best_adj[u, v] = 0
                    best_adj[v, u] = 0
                    best_adj[x, y] = 1
                    best_adj[y, x] = 1
                    
                    new_score = get_score(best_adj)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                    else:
                        best_adj[u, v] = 1
                        best_adj[v, u] = 1
                        best_adj[x, y] = 0
                        best_adj[y, x] = 0
    
    # Return result
    edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j] == 1:
                edges_list.append([i, j])
    
    return {'n': n, 'edges': edges_list}
