import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    # 1. Generate all vectors and their indices
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
    # neighbors_mask[i] is an integer where bit j is 1 if i and j conflict.
    
    neighbors_mask = [0] * N
    
    supports = list(support_map.keys())
    sup_to_vecs = {s: support_map[s] for s in supports}
    
    # Precompute conflict masks
    for i_s, s1 in enumerate(supports):
        for i_t, s2 in enumerate(supports):
            if i_t <= i_s: continue # Symmetry
            inter = set(s1) & set(s2)
            if len(inter) == 3:
                pos_in_s1 = [s1.index(x) for x in inter]
                pos_in_s2 = [s2.index(x) for x in inter]
                
                vecs_s1 = sup_to_vecs[s1]
                vecs_s2 = sup_to_vecs[s2]
                
                for idx1 in vecs_s1:
                    signs1 = all_vectors[idx1][2]
                    mask1 = 0
                    for idx2 in vecs_s2:
                        signs2 = all_vectors[idx2][2]
                        match = True
                        for p1, p2 in zip(pos_in_s1, pos_in_s2):
                            if signs1[p1] != signs2[p2]:
                                match = False
                                break
                        if match:
                            mask1 |= (1 << idx2)
                    neighbors_mask[idx1] |= mask1
                    # Symmetry
                    mask2 = 0
                    # We already added idx1 to idx2's mask? No, we are iterating idx1 in s1, idx2 in s2.
                    # We need to update neighbors_mask[idx2] with idx1.
                    # Since we are inside the loop for idx2, we can just set the bit.
                    # But wait, we are building mask1 for idx1.
                    # We need to also update neighbors_mask[idx2] to include idx1.
                    # It is more efficient to do this in a nested loop or accumulate.
                    # Let's just set the bit directly.
                    neighbors_mask[idx2] |= (1 << idx1)

    # Ensure symmetry is fully handled if the above logic missed any direction due to loop structure
    # The above loop iterates s1, s2. For each pair, it checks all sign combos.
    # If idx1 in s1 conflicts with idx2 in s2, it sets bit idx2 in mask[idx1] and bit idx1 in mask[idx2].
    # This covers all pairs.
    
    # 3. Greedy Initialization for Lower Bound
    def greedy_mis(n, neighbors_mask):
        # Returns a bitmask of a large independent set
        remaining = (1 << n) - 1
        selected = 0
        while remaining:
            # Pick vertex with max degree in remaining subgraph
            best_v = -1
            best_deg = -1
            temp_rem = remaining
            while temp_rem:
                lsb = temp_rem & -temp_rem
                v = lsb.bit_length() - 1
                # Degree in remaining: count bits in (neighbors_mask[v] & remaining)
                deg = (neighbors_mask[v] & remaining).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp_rem ^= lsb
            
            if best_v == -1:
                break
            
            selected |= (1 << best_v)
            # Remove best_v and its neighbors from remaining
            remaining &= ~(neighbors_mask[best_v] | (1 << best_v))
        
        return selected

    start_time = time.time()
    time_limit = 110.0
    
    initial_greedy = greedy_mis(N, neighbors_mask)
    best_size = initial_greedy.bit_count()
    best_mask = initial_greedy
    
    # 4. Branch and Bound MIS
    # We use a recursive function that takes:
    #   P: bitmask of candidate vertices
    #   R: bitmask of vertices already selected (or just size, but mask helps for debugging/extension)
    #   current_size: size of R
    
    sys.setrecursionlimit(10000)
    
    def solve_mis(P, current_size):
        nonlocal best_size, best_mask, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if P == 0:
            if current_size > best_size:
                best_size = current_size
                # Reconstruct mask? We don't track R mask in this signature easily without passing it.
                # Let's pass R mask as well.
            return
        
        # Upper bound: current_size + popcount(P)
        if current_size + P.bit_count() <= best_size:
            return
            
        # Heuristic: Order vertices in P by some criterion
        # Simple heuristic: Just pick the first available vertex? 
        # Better: Pick vertex with max degree in P to prune faster when including it?
        # Or min degree? 
        # Tomita's algorithm uses a specific ordering. 
        # Let's try picking the vertex with the highest degree in the induced subgraph P.
        
        v = -1
        max_deg = -1
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            u = lsb.bit_length() - 1
            deg = (neighbors_mask[u] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp_P ^= lsb
        
        if v == -1:
            return
            
        # Branch 1: Include v
        # New P: P \ {v} \ N(v)
        P_new_1 = P & ~(neighbors_mask[v] | (1 << v))
        solve_mis(P_new_1, current_size + 1)
        
        # Branch 2: Exclude v
        # New P: P \ {v}
        P_new_2 = P & ~(1 << v)
        solve_mis(P_new_2, current_size)

    # We need to track the actual mask for the best solution to return it.
    # Modifying solve_mis to track the mask is tricky with recursion state.
    # Alternative: Store the best mask globally when a new best is found.
    # But we need the mask of selected vertices.
    # Let's change the signature to pass the current selected mask.
    
    def solve_mis_v2(P, R, current_size):
        nonlocal best_size, best_mask, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if P == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = R
            return
        
        if current_size + P.bit_count() <= best_size:
            return
            
        # Select vertex v from P
        # Heuristic: Max degree in P
        v = -1
        max_deg = -1
        temp_P = P
        while temp_P:
            lsb = temp_P & -temp_P
            u = lsb.bit_length() - 1
            deg = (neighbors_mask[u] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
            temp_P ^= lsb
        
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Branch 1: Include v
        R_new = R | bit_v
        P_new = P & ~(neighbors_mask[v] | bit_v)
        solve_mis_v2(P_new, R_new, current_size + 1)
        
        # Branch 2: Exclude v
        P_new = P & ~bit_v
        solve_mis_v2(P_new, R, current_size)

    try:
        # Start with empty R and all vertices in P
        # To improve pruning, we can start with the greedy solution?
        # No, standard MIS search starts with empty set. 
        # However, we can use the greedy result to set the initial best_size.
        # The search will try to beat it.
        
        # Optimization: We can order the initial vertices or use a different start.
        # Let's just run the search.
        solve_mis_v2((1 << N) - 1, 0, 0)
        
    except RecursionError:
        pass

    # Convert best_mask to array
    result_vectors = []
    temp_mask = best_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        result_vectors.append(all_vectors[idx][3])
        temp_mask ^= lsb
        
    if not result_vectors:
        return np.array([], dtype=np.int64).reshape(0, D)
        
    return np.array(result_vectors, dtype=np.int64)
