import time
import random
import sys
import heapq

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Generate all edges and map them to indices
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    # 2. Generate all squares (2D faces)
    square_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, u), max(v2, u))
                
                # Get indices
                idxs = []
                for e in (e1, e2, e3, e4):
                    if e not in edge_index:
                        # Should not happen
                        continue
                    idxs.append(edge_index[e])
                
                if len(idxs) == 4:
                    square_set.add(frozenset(idxs))
    
    squares = [list(s) for s in square_set]
    squares_t = [tuple(sq) for sq in squares]
    
    # 3. Build incidence lists
    edge_squares = [[] for _ in range(n_edges)]
    for sq_idx, sq in enumerate(squares_t):
        for e_idx in sq:
            edge_squares[e_idx].append(sq_idx)

    def time_exceeded():
        return (time.time() - start_time) > time_limit

    def run_local_search(initial_set):
        in_set = [False] * n_edges
        for e in initial_set:
            in_set[e] = True
        
        blockers = [0] * n_edges
        for sq_idx, sq in enumerate(squares_t):
            count_in = sum(1 for e in sq if in_set[e])
            if count_in == 3:
                for e in sq:
                    if not in_set[e]:
                        blockers[e] += 1
        
        current_size = len(initial_set)
        
        def add_all_possible():
            nonlocal in_set, blockers, current_size
            heap = [i for i in range(n_edges) if not in_set[i] and blockers[i] == 0]
            heapq.heapify(heap)
            while heap:
                if time_exceeded():
                    break
                i = heapq.heappop(heap)
                if in_set[i] or blockers[i] != 0:
                    continue
                in_set[i] = True
                current_size += 1
                for sq_idx in edge_squares[i]:
                    sq = squares_t[sq_idx]
                    for j in sq:
                        if j == i or in_set[j]:
                            continue
                        count_other = sum(1 for k in sq if k not in (i, j) and in_set[k])
                        if count_other == 2:
                            blockers[j] += 1
        
        add_all_possible()
        best_size = current_size
        best_state = in_set[:]
        
        edges_to_try = list(range(n_edges))
        while not time_exceeded():
            random.shuffle(edges_to_try)
            improvement_found = False
            for e in edges_to_try:
                if time_exceeded():
                    break
                if not in_set[e]:
                    continue
                
                saved_in = in_set[:]
                saved_blk = blockers[:]
                saved_sz = current_size
                
                in_set[e] = False
                current_size -= 1
                for sq_idx in edge_squares[e]:
                    sq = squares_t[sq_idx]
                    for j in sq:
                        if j == e or in_set[j]:
                            continue
                        count_other = sum(1 for k in sq if k not in (e, j) and in_set[k])
                        if count_other == 2:
                            blockers[j] -= 1
                
                add_all_possible()
                
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    improvement_found = True
                else:
                    in_set = saved_in
                    blockers = saved_blk
                    current_size = saved_sz
            
            if not improvement_found:
                break
        
        return [i for i in range(n_edges) if best_state[i]]

    best_edges_global = []
    best_size_global = 0
    
    # Structured initial solutions: start with one full dimension
    for dim_start in range(d):
        if time_exceeded():
            break
        init_set = []
        in_set = [False] * n_edges
        for u in range(n_vertices):
            v = u ^ (1 << dim_start)
            if u < v:
                e_idx = edge_index[(u, v)]
                in_set[e_idx] = True
                init_set.append(e_idx)
        remaining = [i for i in range(n_edges) if not in_set[i]]
        random.shuffle(remaining)
        for e in remaining:
            ok = True
            for sq_idx in edge_squares[e]:
                sq = squares_t[sq_idx]
                count_in = sum(1 for k in sq if k != e and in_set[k])
                if count_in == 3:
                    ok = False
                    break
            if ok:
                in_set[e] = True
                init_set.append(e)
        
        result = run_local_search(init_set)
        if len(result) > best_size_global:
            best_size_global = len(result)
            best_edges_global = result
    
    # Random restarts
    while not time_exceeded():
        random_edges = list(range(n_edges))
        random.shuffle(random_edges)
        init_set = []
        in_set = [False] * n_edges
        for e in random_edges:
            ok = True
            for sq_idx in edge_squares[e]:
                sq = squares_t[sq_idx]
                count_in = sum(1 for k in sq if k != e and in_set[k])
                if count_in == 3:
                    ok = False
                    break
            if ok:
                in_set[e] = True
                init_set.append(e)
        
        result = run_local_search(init_set)
        if len(result) > best_size_global:
            best_size_global = len(result)
            best_edges_global = result
        
        if best_size_global >= 308:
            break

    final_edges = []
    for e_idx in best_edges_global:
        u, v = edges[e_idx]
        final_edges.append([u, v])
    return final_edges
