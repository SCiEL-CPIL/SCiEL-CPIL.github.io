import numpy as np
import itertools
import time

D = 11

def construct():
    # 1. Generate all vectors
    all_vectors = []
    idx = 0
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in zip(comb, signs):
                v[k] = s
            all_vectors.append(v)
            idx += 1
    N = len(all_vectors)
    
    # 2. Build Conflict Graph as Bitsets
    # neighbors[i] is an integer where bit j is 1 if i conflicts with j
    neighbors = [0] * N
    
    # Group by support to speed up conflict detection
    support_map = {}
    for i, v in enumerate(all_vectors):
        sup = tuple(i for i in range(D) if v[i] != 0)
        if sup not in support_map:
            support_map[sup] = []
        support_map[sup].append(i)
        
    supports = list(support_map.keys())
    
    for i_s in range(len(supports)):
        s1 = supports[i_s]
        vecs1 = support_map[s1]
        for i_t in range(i_s + 1, len(supports)):
            s2 = supports[i_t]
            inter = set(s1) & set(s2)
            if len(inter) != 3:
                continue
            
            pos_in_s1 = [s1.index(x) for x in inter]
            pos_in_s2 = [s2.index(x) for x in inter]
            
            vecs2 = support_map[s2]
            
            for v1_idx in vecs1:
                v1 = all_vectors[v1_idx]
                sig1 = [v1[x] for x in inter]
                
                for v2_idx in vecs2:
                    v2 = all_vectors[v2_idx]
                    sig2 = [v2[x] for x in inter]
                    
                    if sig1 == sig2:
                        neighbors[v1_idx] |= (1 << v2_idx)
                        neighbors[v2_idx] |= (1 << v1_idx)
                        
    # 3. Solve MIS using Tomita's algorithm with bitsets
    best_set = []
    start_time = time.time()
    time_limit = 110.0
    
    # Greedy initialization for lower bound
    cand_mask = (1 << N) - 1
    greedy_sel = []
    while cand_mask:
        lsb = cand_mask & -cand_mask
        v = lsb.bit_length() - 1
        greedy_sel.append(v)
        cand_mask &= ~(neighbors[v] | (1 << v))
        
    best_size = len(greedy_sel)
    best_set = list(greedy_sel)
    
    def greedy_coloring(candidates):
        colors = []
        uncolored = candidates
        while uncolored:
            avail = uncolored
            is_mask = 0
            while avail:
                lsb = avail & -avail
                u = lsb.bit_length() - 1
                is_mask |= lsb
                avail &= ~(neighbors[u] | lsb)
            colors.append(is_mask)
            uncolored &= ~is_mask
        return colors

    def expand(candidates, selected):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = list(selected)
            return
        
        # Quick bound
        if len(selected) + candidates.bit_count() <= best_size:
            return
            
        # Coloring bound
        colors = greedy_coloring(candidates)
        if len(selected) + len(colors) <= best_size:
            return
            
        # MCV Heuristic: Pick vertex with max degree in candidates
        max_deg = -1
        v = -1
        c = candidates
        while c:
            lsb = c & -c
            u = lsb.bit_length() - 1
            deg = (candidates & neighbors[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            c ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        new_cands = candidates & ~(neighbors[v] | (1 << v))
        selected.append(v)
        expand(new_cands, selected)
        selected.pop()
        
        # Branch 2: Exclude v
        new_cands2 = candidates & ~(1 << v)
        expand(new_cands2, selected)
        
    expand((1 << N) - 1, [])
    
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    result_vectors = [all_vectors[i] for i in best_set]
    return np.array(result_vectors, dtype=np.int64)
