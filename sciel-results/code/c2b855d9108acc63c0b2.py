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
    
    # 2. Build Conflict Graph using Bitmasks
    # adj[i] is a bitmask of vertices conflicting with i
    
    adj = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Map support tuple to a list of (signs_tuple, idx)
    # We need to check conflicts between supports with intersection size 3.
    
    # Precompute for each support, the vectors and their sign patterns
    # signs are encoded as 4-bit integers for fast comparison?
    # Or just keep as tuples.
    
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    # Create a signature of signs on intersection for s1
                    # We need to compare signs1 at pos_in_s1 with signs2 at pos_in_s2
                    sig1 = tuple(signs1[p] for p in pos_in_s1)
                    
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        sig2 = tuple(signs2[p] for p in pos_in_s2)
                        
                        if sig1 == sig2:
                            # Conflict
                            adj[idx1] |= (1 << idx2)
                            adj[idx2] |= (1 << idx1)

    # 3. Solve MIS using Bitset Branch and Bound
    
    best_size = 0
    best_mask = 0
    
    # Helper: Popcount
    def popcount(x):
        return bin(x).count('1')
    
    # Helper: Greedy Lower Bound (Maximal Independent Set)
    # Returns a mask of a maximal independent set and its size
    def greedy_mis(candidates):
        selected = 0
        remaining = candidates
        while remaining:
            # Pick vertex with min degree in remaining? Or just first?
            # For speed, just pick the lowest set bit?
            # Better: pick vertex with min degree to maximize size?
            # Let's just pick the first one for speed in the bound calculation.
            # Actually, a better greedy is to pick min degree.
            # But min degree requires iteration.
            # Let's try picking the vertex with the fewest neighbors in `remaining`.
            # This is O(V) per step, total O(V^2). V=5280. 28M ops. Might be slow if called often.
            # Alternative: Just pick the lowest bit. This is fast but yields smaller sets.
            # Let's use a simple heuristic: pick the vertex with min degree in the *induced* subgraph.
            # We can approximate degree by popcount(adj[v] & remaining).
            
            # To make this fast, we'll just iterate and find min degree.
            min_deg = float('inf')
            v = -1
            temp = remaining
            # Iterate through set bits
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    v = u
                temp ^= lsb
                
            if v == -1:
                break
                
            selected |= (1 << v)
            # Remove v and its neighbors from remaining
            remaining &= ~(adj[v] | (1 << v))
            
        return selected, popcount(selected)

    start_time = time.time()
    time_limit = 115.0

    def solve(candidates, current_mask, current_count):
        nonlocal best_size, best_mask, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
            
        if not candidates:
            if current_count > best_size:
                best_size = current_count
                best_mask = current_mask
            return
            
        # Upper bound: current_count + popcount(candidates)
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Try to improve best_size with a greedy solution on candidates
        # This helps prune earlier
        g_mask, g_count = greedy_mis(candidates)
        if current_count + g_count > best_size:
            best_size = current_count + g_count
            best_mask = current_mask | g_mask
            
        # If even the greedy didn't beat best, and the trivial bound is tight?
        # The trivial bound check was already done.
        # But greedy might have updated best_size.
        if current_count + popcount(candidates) <= best_size:
            return
            
        # Heuristic: Pick vertex v with max degree in candidates
        # This reduces the branching factor significantly.
        max_deg = -1
        v = -1
        temp = candidates
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(adj[u] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v == -1:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New candidates: candidates without v and its neighbors
        new_candidates_1 = candidates & ~(adj[v] | v_bit)
        solve(new_candidates_1, current_mask | v_bit, current_count + 1)
        
        # Branch 2: Exclude v
        # New candidates: candidates without v
        new_candidates_2 = candidates & ~v_bit
        solve(new_candidates_2, current_mask, current_count)

    initial_candidates = (1 << N) - 1
    solve(initial_candidates, 0, 0)
    
    # Convert best_mask to array
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
