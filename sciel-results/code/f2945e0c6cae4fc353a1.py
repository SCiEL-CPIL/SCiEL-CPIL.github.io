import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10 sec margin
    
    # 1. Precompute all k-blocks
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Convert to bitmasks for fast intersection
    block_masks = []
    for b in all_blocks:
        mask = 0
        for p in b:
            mask |= (1 << p)
        block_masks.append(mask)
        
    # 2. Precompute binomial coefficients for C(m, t) where m is intersection size
    def nCr(n, r):
        if r < 0 or r > n:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        res = 1
        for i in range(r):
            res = res * (n - i) // (i + 1)
        return res
            
    # We need C(m, t) for m from 0 to k
    binom_table = [nCr(m, t) for m in range(k + 1)]
    initial_score = nCr(k, t)

    def greedy_on_indices(masks):
        n = len(masks)
        scores = [initial_score] * n
        used = [False] * n
        selected_indices = []
        
        for _ in range(n):
            max_score = -1
            max_idx = -1
            for i in range(n):
                if not used[i] and scores[i] > max_score:
                    max_score = scores[i]
                    max_idx = i
            if max_idx == -1:
                break
            used[max_idx] = True
            selected_indices.append(max_idx)
            mask_i = masks[max_idx]
            for j in range(n):
                if not used[j]:
                    m = (mask_i & masks[j]).bit_count()
                    # binom_table[m] is 0 if m < t
                    scores[j] -= binom_table[m]
        return selected_indices

    def remove_redundant(blocks):
        if not blocks:
            return blocks
            
        block_subsets = []
        for b in blocks:
            subs = list(itertools.combinations(b, t))
            block_subsets.append(subs)
            
        count = {}
        for subs in block_subsets:
            for s in subs:
                count[s] = count.get(s, 0) + 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_block_subsets = []
            for i, b in enumerate(blocks):
                redundant = True
                for s in block_subsets[i]:
                    if count[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[i]:
                        count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
                    new_block_subsets.append(block_subsets[i])
            blocks = new_blocks
            block_subsets = new_block_subsets
        return blocks

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        shuffled_indices = list(range(n_blocks))
        random.shuffle(shuffled_indices)
        
        permuted_masks = [block_masks[i] for i in shuffled_indices]
        
        sel_indices = greedy_on_indices(permuted_masks)
        
        # Map back to original block tuples
        # sel_indices are indices into permuted_masks, which correspond to shuffled_indices
        # So original index is shuffled_indices[sel_idx]
        # And block is all_blocks[original_index]
        
        final_blocks = [all_blocks[shuffled_indices[i]] for i in sel_indices]
        
        final_blocks = remove_redundant(final_blocks)
        
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_blocks = final_blocks
            
    if best_blocks is None:
        # Fallback
        sel_indices = greedy_on_indices(block_masks)
        return [all_blocks[i] for i in sel_indices]
        
    return best_blocks
