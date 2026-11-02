import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    # Edge cases
    if k < t:
        return []
    if t == 0:
        return [[i for i in range(v)][:k]] if k > 0 else []
    if v <= k:
        return [list(range(v))]
        
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all (t-1)-subsets
    all_tm1_subsets = list(itertools.combinations(range(v), t-1)) if t > 1 else [()]
    N_tm1 = len(all_tm1_subsets)
    tm1_subset_to_idx = {s: i for i, s in enumerate(all_tm1_subsets)}
    
    # ext_idx[x][j] = index of t-subset formed by x and tm1-subset j, or -1 if x in tm1-subset j
    # Dimensions: v x N_tm1
    ext_idx = [[-1] * N_tm1 for _ in range(v)]
    for x in range(v):
        for j, tm1_s in enumerate(all_tm1_subsets):
            if x in tm1_s:
                continue
            t_s = tuple(sorted(tm1_s + (x,)))
            ext_idx[x][j] = t_subset_to_idx[t_s]

    # Greedy Construction
    uncovered = (1 << N_t) - 1
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Find the first uncovered subset (lowest set bit)
        # This is O(1) if we use bit operations, but finding the index requires log(N) or similar
        # Python's int.bit_length is fast.
        # To find the index of the lowest set bit:
        # (uncovered & -uncovered).bit_length() - 1
        low_bit = uncovered & -uncovered
        seed_idx = low_bit.bit_length() - 1
        seed_subset = all_t_subsets[seed_idx]
        
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = -1
            best_gain = -1
            
            # Precompute indices of (t-1) subsets of current block
            block_tm1_indices = []
            if t > 1:
                for tm1_comb in itertools.combinations(block, t-1):
                    block_tm1_indices.append(tm1_subset_to_idx[tm1_comb])
            else:
                # t=1, block has 0 elements, we just add 1 point
                block_tm1_indices = [0] # dummy, logic below handles t=1 separately? 
                # Actually if t=1, gain is 1 if x not covered.
                
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                if t == 1:
                    # Check if x is uncovered
                    idx = t_subset_to_idx[(x,)]
                    if (uncovered >> idx) & 1:
                        gain = 1
                else:
                    for j in block_tm1_indices:
                        idx_t = ext_idx[x][j]
                        if idx_t != -1 and ((uncovered >> idx_t) & 1):
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered mask
        for comb in itertools.combinations(block, t):
            idx = t_subset_to_idx[comb]
            uncovered &= ~(1 << idx)

    # Pruning Phase
    # Count how many blocks cover each t-subset
    counts = [0] * N_t
    for b in blocks:
        for comb in itertools.combinations(b, t):
            counts[t_subset_to_idx[comb]] += 1
            
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        new_blocks = []
        for b in blocks:
            redundant = True
            for comb in itertools.combinations(b, t):
                if counts[t_subset_to_idx[comb]] == 1:
                    redundant = False
                    break
            
            if not redundant:
                # Remove b
                for comb in itertools.combinations(b, t):
                    counts[t_subset_to_idx[comb]] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    return blocks
