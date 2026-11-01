import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    k_covers = [[] for _ in range(num_k)]
    k_masks = [0] * num_k
    
    for i, ks in enumerate(k_subsets):
        covers = []
        m = 0
        for ts in itertools.combinations(ks, t):
            idx = t_to_idx[ts]
            covers.append(idx)
            m |= (1 << idx)
        k_covers[i] = covers
        k_masks[i] = m
        
    # Precompute sets for fast swap validation
    k_covers_sets = [set(c) for c in k_covers]

    def init_greedy():
        uncovered = (1 << num_t) - 1
        blocks = []
        
        while uncovered:
            best_gain = 0
            best_idx = -1
            # Randomize tie-breaking to get diverse solutions
            # We can just pick the first max, or shuffle indices?
            # Shuffling 75k indices every step is too slow.
            # Instead, we can collect candidates with max gain and pick one randomly.
            
            max_gain = 0
            candidates = []
            
            for i in range(num_k):
                gain = (k_masks[i] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            chosen = random.choice(candidates)
            blocks.append(chosen)
            uncovered &= ~k_masks[chosen]
            
        return blocks

    def local_search(initial_blocks):
        blocks = list(initial_blocks)
        cover_counts = [0] * num_t
        
        for b in blocks:
            for idx in k_covers[b]:
                cover_counts[idx] += 1
                
        num_uncovered = 0
        for c in cover_counts:
            if c == 0:
                num_uncovered += 1
                
        if num_uncovered > 0:
            return float('inf'), []
            
        best_len = len(blocks)
        best_blocks = list(blocks)
        
        # Local search
        swaps_per_restart = 2000
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try Deletions
            block_indices = list(range(len(blocks)))
            random.shuffle(block_indices)
            
            for bi in block_indices:
                b = blocks[bi]
                can_delete = True
                for idx in k_covers[b]:
                    if cover_counts[idx] == 1:
                        can_delete = False
                        break
                
                if can_delete:
                    for idx in k_covers[b]:
                        cover_counts[idx] -= 1
                    blocks.pop(bi)
                    improved = True
                    break
                    
            if improved:
                continue
                
            # 2. Try Swaps
            found_swap = False
            for _ in range(swaps_per_restart):
                if time.time() > deadline:
                    break
                
                if not blocks:
                    break
                    
                old_bi = random.randint(0, len(blocks) - 1)
                old_b = blocks[old_bi]
                
                new_b = random.randint(0, num_k - 1)
                if new_b == old_b:
                    continue
                    
                # Check validity
                uncovered_by_old = []
                for idx in k_covers[old_b]:
                    cover_counts[idx] -= 1
                    if cover_counts[idx] == 0:
                        uncovered_by_old.append(idx)
                        
                if uncovered_by_old:
                    # Check if new_b covers all uncovered_by_old
                    new_set = k_covers_sets[new_b]
                    valid_swap = all(idx in new_set for idx in uncovered_by_old)
                else:
                    valid_swap = True
                    
                if valid_swap:
                    # Perform swap
                    for idx in k_covers[new_b]:
                        cover_counts[idx] += 1
                    blocks[old_bi] = new_b
                    found_swap = True
                    break
                else:
                    # Revert
                    for idx in k_covers[old_b]:
                        cover_counts[idx] += 1
                        
            if found_swap:
                continue
            else:
                # Stuck
                break
                
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        return best_len, best_blocks

    best_count = float('inf')
    best_blocks_list = None
    
    while time.time() < deadline:
        greedy_blocks = init_greedy()
        count, blocks = local_search(greedy_blocks)
        
        if count < best_count:
            best_count = count
            best_blocks_list = [list(k_subsets[i]) for i in blocks]
            
    if best_blocks_list is None:
        # Fallback
        greedy_blocks = init_greedy()
        best_blocks_list = [list(k_subsets[i]) for i in greedy_blocks]
        
    return best_blocks_list
