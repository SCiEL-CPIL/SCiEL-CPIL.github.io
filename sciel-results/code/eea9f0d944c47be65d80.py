import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_subs = list(itertools.combinations(range(v), k))
    num_k_subs = len(k_subs)
    
    block_masks = [0] * num_k_subs
    for i, block in enumerate(k_subs):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    max_possible = math.comb(k, t)
    lower_bound = math.ceil(num_t_subs / max_possible)
    
    best_blocks = None
    best_len = float('inf')
    
    # Helper to run one greedy instance
    def run_greedy():
        uncovered = (1 << num_t_subs) - 1
        current_blocks = []
        indices = list(range(num_k_subs))
        random.shuffle(indices)
        
        while uncovered:
            max_gain = 0
            best_indices = []
            
            # Local vars for speed
            bm = block_masks
            unc = uncovered
            mp = max_possible
            
            for idx in indices:
                gain = (bm[idx] & unc).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_indices = [idx]
                    if max_gain == mp:
                        break
                elif gain == max_gain and gain > 0:
                    best_indices.append(idx)
            
            if max_gain == 0:
                break
                
            chosen_idx = random.choice(best_indices)
            current_blocks.append(k_subs[chosen_idx])
            uncovered &= ~block_masks[chosen_idx]
            
            # Pruning
            if len(current_blocks) + math.ceil(uncovered.bit_count() / mp) >= best_len:
                break
                
        return current_blocks

    # Run multiple instances
    while time.time() < deadline:
        blocks = run_greedy()
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            if best_len == lower_bound:
                break
                
    return best_blocks
