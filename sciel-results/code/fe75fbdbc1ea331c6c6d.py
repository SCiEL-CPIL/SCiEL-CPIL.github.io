import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors
    all_vectors = []
    support_map = {}
    
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        support = comb
        support_map[support] = []
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(support, signs):
                v[k] = s
            all_vectors.append((idx, support, signs, v))
            support_map[support].append(idx)
            idx += 1
            
    N = len(all_vectors)
    
    # 2. Build Conflict Graph
    adj = [[] for _ in range(N)]
    
    # Precompute sup_vecs
    sup_vecs = {s: [] for s in support_map.keys()}
    for i, (idx, support, signs, v) in enumerate(all_vectors):
        sup_vecs[support].append((signs, idx))
        
    # For each vector u, find conflicts
    for u_idx, u_support, u_signs, u_vec in all_vectors:
        for c1 in range(4):
            for c2 in range(c1+1, 4):
                for c3 in range(c2+1, 4):
                    inter_coords = (u_support[c1], u_support[c2], u_support[c3])
                    sig_u_inter = tuple(u_signs[x] for x in [c1, c2, c3])
                    
                    for x in range(D):
                        if x in u_support:
                            continue
                        
                        s_v_list = list(inter_coords)
                        s_v_list.append(x)
                        s_v_list.sort()
                        s_v = tuple(s_v_list)
                        
                        if s_v not in sup_vecs:
                            continue
                            
                        idxs_in_sv = [s_v.index(c) for c in inter_coords]
                        
                        for v_signs, v_idx in sup_vecs[s_v]:
                            match = True
                            for pos, sign in zip(idxs_in_sv, sig_u_inter):
                                if v_signs[pos] != sign:
                                    match = False
                                    break
                            
                            if match and u_idx != v_idx:
                                adj[u_idx].append(v_idx)
                                adj[v_idx].append(u_idx)

    # Convert to bitmask adjacency
    adj_mask = [0] * N
    for i in range(N):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        adj_mask[i] = m
        
    def popcount(x):
        return bin(x).count('1')
        
    start_time = time.time()
    time_limit = 115.0

    best_size = 0
    best_mask = 0

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        # Upper Bound
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Lower Bound Update (Greedy MIS)
        g_mask = 0
        remaining = candidates
        while remaining:
            lsb = remaining & -remaining
            v = lsb.bit_length() - 1
            g_mask |= (1 << v)
            remaining &= ~(adj_mask[v] | (1 << v))
            
        g_count = popcount(g_mask)
        if current_count + g_count > best_size:
            best_size = current_count + g_count
            best_mask = current_mask | g_mask
            
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Branching
        # Pick vertex with max degree in candidates
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(adj_mask[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        new_candidates_1 = candidates & ~(adj_mask[v] | v_bit)
        solve(new_candidates_1, current_mask | v_bit, current_count + 1)
        
        # Branch 2: Exclude v
        new_candidates_2 = candidates & ~v_bit
        solve(new_candidates_2, current_mask, current_count)

    initial_candidates = (1 << N) - 1
    solve(initial_candidates, 0, 0)
    
    if best_size == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        result_vectors.append(all_vectors[i][3])
        temp ^= lsb
        
    return np.array(result_vectors, dtype=np.int64)
