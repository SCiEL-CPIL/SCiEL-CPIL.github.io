import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def check_full_validity(adj, n):
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                S_arr = np.array(S)
                sub_adj = adj[np.ix_(S_arr, S_arr)]
                degs = sub_adj.sum(axis=1)
                if np.all(degs == degs[0]):
                    return False
        return True

    def to_result(n, adj):
        edges = [[int(u), int(v)] for u in range(n) for v in range(u + 1, n) if adj[u, v] == 1]
        return {'n': int(n), 'edges': edges}

    def greedy_construct(target_n):
        adj = np.zeros((1, 1), dtype=np.uint8)
        
        for k in range(1, target_n):
            n_old = k
            # Precompute data for all 5-subsets of {0, ..., n_old-1}
            # Data: (S_tuple, D, list of (i, D-di) pairs required)
            # If sum_d % 4 != 0, it can never be regular, so skip.
            subset_data = []
            for S in itertools.combinations(range(n_old), 5):
                S_list = list(S)
                sum_d = 0
                ds = []
                for i in S_list:
                    di = 0
                    for j in S_list:
                        if i != j:
                            di += adj[i, j]
                    ds.append(di)
                    sum_d += di
                
                if sum_d % 4 != 0:
                    continue
                
                D = sum_d // 4
                # Check if d_i in {D, D-1}
                valid_d = True
                req_edges = []
                for idx, i in enumerate(S_list):
                    di = ds[idx]
                    if di != D and di != D - 1:
                        valid_d = False
                        break
                    req_edges.append((i, D - di))
                
                if valid_d:
                    subset_data.append((S_list, req_edges))
            
            # Local search on edges
            best_edges = np.random.randint(0, 2, n_old)
            best_count = float('inf')
            
            def count_bad(edges):
                count = 0
                for S_list, req_edges in subset_data:
                    # Check if edges match req_edges
                    match = True
                    for i, req_val in req_edges:
                        if edges[i] != req_val:
                            match = False
                            break
                    if match:
                        count += 1
                return count

            cur_edges = best_edges.copy()
            cur_count = count_bad(cur_edges)
            best_count = cur_count
            
            T = 5.0
            cooling = 0.99
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() > deadline - 10:
                    break
                
                idx = np.random.randint(n_old)
                cur_edges[idx] ^= 1
                new_count = count_bad(cur_edges)
                delta = new_count - cur_count
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    cur_count = new_count
                    if cur_count < best_count:
                        best_count = cur_count
                        best_edges = cur_edges.copy()
                        if best_count == 0:
                            break
                else:
                    cur_edges[idx] ^= 1
                
                T *= cooling
            
            # Extend adj
            new_adj = np.zeros((k + 1, k + 1), dtype=np.uint8)
            new_adj[:k, :k] = adj
            for i in range(k):
                if best_edges[i] == 1:
                    new_adj[k, i] = 1
                    new_adj[i, k] = 1
            adj = new_adj
            
        return adj

    # Try 21
    if time.time() < deadline - 20:
        adj21 = greedy_construct(21)
        if check_full_validity(adj21, 21):
            return to_result(21, adj21)

    # Try 20
    if time.time() < deadline - 10:
        adj20 = greedy_construct(20)
        if check_full_validity(adj20, 20):
            return to_result(20, adj20)

    # Fallback random 20
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        adj = (np.random.rand(20, 20) < 0.5).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        adj = (adj | adj.T).astype(np.uint8)
        if check_full_validity(adj, 20):
            return to_result(20, adj)

    # Fallback 5
    n = 5
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = [[int(u), int(v)] for u in range(5) for v in range(u + 1, 5)]
    return {'n': 5, 'edges': edges}
