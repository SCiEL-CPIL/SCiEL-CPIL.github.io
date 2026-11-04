import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Use supports of size 5
    support_size = 5
    supports = list(itertools.combinations(range(D), support_size))
    N_S = len(supports) # 462
    
    # Convert supports to bitmasks
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # Build conflict graph
    # Conflict if intersection size is 3 or more (conservative for fixed sign patterns)
    # Actually, for size 5, if we use even parity code (16 vectors), min dist in code is 2.
    # Max dot in code is 2 (for size 5, 5-2*2=1? No. 5-2*2 = 1. Wait.
    # d_H=2 -> dot = 5 - 4 = 1.
    # d_H=4 -> dot = 5 - 8 = -3.
    # Max dot is 1.
    # We need dot <= 2.5 (for norm sqrt(5), dist sqrt(5) -> dot <= 2.5 -> dot <= 2).
    # So dot <= 1 is always safe.
    # So within a support, all 16 even-parity vectors are mutually valid.
    
    # Between supports S1, S2:
    # We need max_{u in Code(S1), v in Code(S2)} u.v <= 2.
    # If intersection is 0, 1, 2:
    # Max dot is intersection_size (if signs align).
    # If inter=2, max dot 2. 2 <= 2. OK.
    # If inter=3, max dot 3. 3 > 2. FAIL.
    # So we must avoid intersection 3 if we use "all" even parity vectors.
    # Thus, Conflict if intersection >= 3.
    
    adj_mask = [0] * N_S
    
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter >= 3:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # Maximum Independent Set
    best_size = 0
    best_set_mask = 0
    
    def greedy_init():
        remaining = (1 << N_S) - 1
        s_mask = 0
        while remaining:
            v_bit = remaining & -remaining
            v_idx = v_bit.bit_length() - 1
            s_mask |= v_bit
            remaining &= ~(adj_mask[v_idx] | v_bit)
        return s_mask

    init_set = greedy_init()
    best_size = init_set.bit_count()
    best_set_mask = init_set
    
    def color_bound(candidates):
        if candidates == 0:
            return 0, []
        
        colors = 0
        color_classes = []
        remaining = candidates
        
        while remaining:
            colors += 1
            color_class = 0
            temp = remaining
            while temp:
                v_bit = temp & -temp
                v_idx = v_bit.bit_length() - 1
                color_class |= v_bit
                temp &= ~(adj_mask[v_idx] | v_bit)
            
            color_classes.append(color_class)
            remaining &= ~color_class
        
        return colors, color_classes
    
    def branch_and_bound(candidates, current_set, current_size):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline - 2.0:
            return
            
        ub, color_classes = color_bound(candidates)
        
        if current_size + ub <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_set
            return
            
        # Pivot selection
        v_bit = candidates & -candidates
        v_idx = v_bit.bit_length() - 1
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
        new_candidates = candidates & ~(adj_mask[pivot_idx] | pivot_bit)
        branch_and_bound(new_candidates, current_set | pivot_bit, current_size + 1)
        branch_and_bound(candidates & ~pivot_bit, current_set, current_size)

    branch_and_bound((1 << N_S) - 1, 0, 0)

    # Construct vectors
    # For each support, add all 16 even-parity sign vectors
    # Even parity: number of -1s is even.
    # This ensures min Hamming distance 2 within the support.
    # Max dot product within support: 1 (for d_H=2). 1 <= 2. OK.
    
    vectors = []
    temp = best_set_mask
    while temp:
        lsb = temp & -temp
        s_idx = lsb.bit_length() - 1
        support = supports[s_idx]
        
        # Generate even parity signs for 5 coordinates
        # There are 16 such signs.
        for signs in itertools.product([1, -1], repeat=5):
            # Count -1s
            neg_count = sum(1 for s in signs if s == -1)
            if neg_count % 2 == 0:
                v = np.zeros(D, dtype=np.int8)
                for idx, sign in zip(support, signs):
                    v[idx] = sign
                vectors.append(v)
        temp ^= lsb

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
