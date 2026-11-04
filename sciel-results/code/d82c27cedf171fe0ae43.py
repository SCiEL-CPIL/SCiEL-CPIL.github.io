import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    adj_mask = [0] * N_S
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # 3. Maximum Independent Set using Branch and Bound with Pivot-based Coloring
    # Tomita's algorithm adapted for MIS
    
    best_size = 0
    best_set_mask = 0
    
    # Greedy initialization for lower bound
    def greedy_init():
        remaining = (1 << N_S) - 1
        s_mask = 0
        while remaining:
            v_bit = remaining & -remaining
            v_idx = v_bit.bit_length() - 1
            s_mask |= v_bit
            remaining &= ~(adj_mask[v_idx] | v_bit)
        return s_mask

    # Initial lower bound
    init_set = greedy_init()
    best_size = init_set.bit_count()
    best_set_mask = init_set
    
    # Tomita-style coloring for upper bound
    # We color the subgraph induced by candidates
    # The number of colors is an upper bound on MIS size
    
    def color_bound(candidates):
        """
        Returns (upper_bound, color_classes) where upper_bound is the number of colors
        used in a greedy coloring of the graph induced by candidates.
        """
        if candidates == 0:
            return 0, []
        
        colors = 0
        color_classes = []
        remaining = candidates
        
        while remaining:
            colors += 1
            # Greedy: find all vertices in remaining that don't conflict with each other
            # This is itself a greedy independent set
            color_class = 0
            temp = remaining
            while temp:
                v_bit = temp & -temp
                v_idx = v_bit.bit_length() - 1
                color_class |= v_bit
                # Remove v and its neighbors from temp (within remaining)
                temp &= ~(adj_mask[v_idx] | v_bit)
            
            color_classes.append(color_class)
            remaining &= ~color_class
        
        return colors, color_classes
    
    # Branch and bound with pivot selection
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline - 2.0:
            return
            
        # Get upper bound via coloring
        ub, color_classes = color_bound(candidates)
        
        if current_size + ub <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set
            return
            
        # Pivot selection: choose vertex with highest degree in candidates
        # to maximize pruning
        v_bit = candidates & -candidates
        v_idx = v_bit.bit_length() - 1
        
        # Find vertex with max degree in candidates
        max_deg = -1
        pivot_bit = v_bit
        pivot_idx = v_idx
        temp = candidates
        while temp:
            b = temp & -temp
            idx = b.bit_length() - 1
            deg = (adj_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot_bit = b
                pivot_idx = idx
            temp ^= b
        
        # Branch on pivot
        # Include pivot
        new_candidates = candidates & ~(adj_mask[pivot_idx] | pivot_bit)
        branch_and_bound(new_candidates, current_set | pivot_bit, current_size + 1)
        
        # Exclude pivot
        branch_and_bound(candidates & ~pivot_bit, current_set, current_size)

    branch_and_bound((1 << N_S) - 1, 0, 0)

    # 4. Construct vectors
    vectors = []
    temp = best_set_mask
    while temp:
        lsb = temp & -temp
        s_idx = lsb.bit_length() - 1
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
        temp ^= lsb

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
