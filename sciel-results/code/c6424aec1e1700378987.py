import itertools
import time
import random

def construct(v, k, t):
    # Specific optimization for v=19, k=8, t=5
    if v != 19 or k != 8 or t != 5:
        # Fallback to a simple greedy for other parameters (not expected)
        return _generic_greedy(v, k, t)
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Precompute indices for all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    total_t = len(t_subsets)
    
    # 2. Precompute masks for all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    masks = []
    for k_sub in k_subsets:
        mask = 0
        # Get all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            mask |= 1 << t_subset_to_idx[t_sub]
        masks.append(mask)
    
    all_ones = (1 << total_t) - 1
    
    def update_counts(counts, mask, delta):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += delta
            m ^= lsb

    def reduce_blocks(blocks):
        if not blocks:
            return blocks
        counts = [0] * total_t
        for b in blocks:
            update_counts(counts, masks[b], 1)
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                removable = True
                m = masks[b]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        removable = False
                        break
                    m ^= lsb
                if removable:
                    update_counts(counts, masks[b], -1)
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks

    def greedy_once(seed):
        random.seed(seed)
        uncovered = all_ones
        selected = []
        
        while uncovered:
            max_gain = 0
            candidates = []
            # Iterate through all blocks to find max gain
            for i, m in enumerate(masks):
                g = (uncovered & m).bit_count()
                if g > max_gain:
                    max_gain = g
                    candidates = [i]
                elif g == max_gain and g > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered is not 0 and we have valid blocks
                
            choice = random.choice(candidates)
            selected.append(choice)
            uncovered &= ~masks[choice]
            
        # Post-process to remove redundant blocks
        selected = reduce_blocks(selected)
        return selected

    best_blocks = None
    best_len = float('inf')
    
    run_id = 0
    while time.time() < deadline:
        blocks = greedy_once(run_id)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # Optional: if we hit the known lower bound or a very good number, we could stop early?
            # But let's try to improve further.
        run_id += 1
        
    if best_blocks is None:
        # Fallback if no runs completed (should not happen)
        return []
        
    # Convert indices back to lists of points
    result = []
    for b_idx in best_blocks:
        result.append(list(k_subsets[b_idx]))
        
    return result

def _generic_greedy(v, k, t):
    # Fallback for other parameters
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
