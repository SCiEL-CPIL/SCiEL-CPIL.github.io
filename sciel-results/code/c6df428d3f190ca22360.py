import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 200 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i]) # Initially all uncovered
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find max gain
            max_gain = 0
            candidates = []
            # Optimization: If we have a heap, we could use it, but gain changes dynamically.
            # Scanning 75k is fast.
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Random choice among max gain candidates
            chosen_idx = random.choice(candidates)
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            # Iterate over set bits in newly_covered
            nc = newly_covered
            while nc:
                # Get lowest set bit
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                
                # Decrement gain for all k-subsets covering t-subset idx_j
                for i in t_to_k[idx_j]:
                    if i not in used_blocks: # Only update unused blocks? 
                        # Actually, if i is used, we don't care about its gain for selection, 
                        # but if we remove it later in local search, we might need it.
                        # For greedy, we only select unused.
                        # So we can skip used blocks to save time? 
                        # But if we do local search, we might swap. 
                        # Let's just update all for consistency, it's fast enough.
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0 # Safety

        if uncovered == 0 and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
        # Local Search Phase
        if best_blocks is not None:
             # Try to improve best_blocks
             # This is complex to implement efficiently in the remaining time.
             # Given the speed of greedy, we can just do more restarts.
             pass

    if best_blocks is None:
        # Fallback
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
