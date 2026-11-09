import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using randomized greedy set cover
    with multiple restarts and local search (1-exchange and redundancy removal).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    block_masks = []
    all_blocks = []
    
    # Generate all k-blocks and their masks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    num_blocks = len(block_masks)
    
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')
    
    full_mask = (1 << num_t_subsets) - 1
    
    def greedy_set_cover(randomize=False, temperature=1.0, rng=None):
        """Run greedy set cover, optionally with randomized selection."""
        if rng is None:
            rng = random
        
        current_uncovered = full_mask
        chosen_indices = []
        chosen_masks = []
        
        while current_uncovered != 0:
            if time.time() > deadline:
                break
            
            # Compute gains for all blocks
            gains = []
            for i in range(num_blocks):
                intersection = block_masks[i] & current_uncovered
                gains.append(count_bits(intersection))
            
            max_gain = max(gains)
            if max_gain == 0:
                break
            
            if randomize:
                # Weighted random selection: probability proportional to gain^temperature
                # Filter to blocks with positive gain
                positive_indices = [i for i in range(num_blocks) if gains[i] > 0]
                weights = [gains[i] ** temperature for i in positive_indices]
                total_weight = sum(weights)
                if total_weight == 0:
                    break
                # Normalize
                probs = [w / total_weight for w in weights]
                # Sample
                r = rng.random()
                cum = 0.0
                selected_idx = positive_indices[-1]
                for idx, p in zip(positive_indices, probs):
                    cum += p
                    if r <= cum:
                        selected_idx = idx
                        break
            else:
                # Deterministic: pick max gain (first if tie)
                selected_idx = 0
                for i in range(num_blocks):
                    if gains[i] > gains[selected_idx]:
                        selected_idx = i
            
            chosen_indices.append(selected_idx)
            chosen_masks.append(block_masks[selected_idx])
            current_uncovered &= ~block_masks[selected_idx]
        
        return chosen_indices, chosen_masks
    
    def remove_redundant(chosen_indices, chosen_masks):
        """Remove blocks whose coverage is fully covered by other chosen blocks."""
        n = len(chosen_indices)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if time.time() > deadline:
                    return chosen_indices, chosen_masks
                
                mask_i = chosen_masks[i]
                
                # Calculate union of all other masks
                union_others = 0
                for j in range(n):
                    if i != j:
                        union_others |= chosen_masks[j]
                
                # Check if mask_i is subset of union_others
                if (mask_i & ~union_others) == 0:
                    chosen_indices.pop(i)
                    chosen_masks.pop(i)
                    n -= 1
                    changed = True
                    break
        
        return chosen_indices, chosen_masks
    
    def local_search_1_exchange(chosen_indices, chosen_masks, max_iterations=50):
        """Try to replace one block with another to reduce total count or improve coverage."""
        n = len(chosen_indices)
        if n == 0:
            return chosen_indices, chosen_masks
        
        # Compute total coverage
        total_cov = 0
        for m in chosen_masks:
            total_cov |= m
        
        # Try removing each block and see if we can cover the lost t-subsets with one new block
        for i in range(n):
            if time.time() > deadline:
                return chosen_indices, chosen_masks
            
            mask_i = chosen_masks[i]
            
            # What would be uncovered if we remove block i?
            union_without_i = 0
            for j in range(n):
                if j != i:
                    union_without_i |= chosen_masks[j]
            
            lost = mask_i & ~union_without_i
            if lost == 0:
                # Block i is redundant, remove it
                chosen_indices.pop(i)
                chosen_masks.pop(i)
                n -= 1
                # Restart to check for more redundancies
                return local_search_1_exchange(chosen_indices, chosen_masks, max_iterations)
            
            # Find a single block that covers all of 'lost'
            # and doesn't add too much overhead
            best_replacement = -1
            best_score = -1
            
            for i_new in range(num_blocks):
                if i_new in chosen_indices:
                    continue
                m_new = block_masks[i_new]
                if (lost & ~m_new) == 0:
                    # This block covers all lost t-subsets
                    # Score: prefer blocks that cover more of the current coverage (less redundant)
                    # and have less overlap with existing blocks
                    new_total = union_without_i | m_new
                    # We want to minimize the number of blocks, so any valid replacement is good
                    # But prefer one that covers more uncovered territory
                    gain = count_bits(m_new & ~union_without_i)
                    if gain > best_score:
                        best_score = gain
                        best_replacement = i_new
            
            if best_replacement != -1:
                # Replace block i with best_replacement
                chosen_indices[i] = best_replacement
                chosen_masks[i] = block_masks[best_replacement]
        
        return chosen_indices, chosen_masks
    
    def try_reduce_by_one(chosen_indices, chosen_masks):
        """Try to reduce block count by 1 via 1-exchange."""
        n = len(chosen_indices)
        if n == 0:
            return chosen_indices, chosen_masks
        
        for i in range(n):
            if time.time() > deadline:
                return chosen_indices, chosen_masks
            
            mask_i = chosen_masks[i]
            
            # Union of all other blocks
            union_others = 0
            for j in range(n):
                if j != i:
                    union_others |= chosen_masks[j]
            
            # t-subsets only covered by block i
            lost = mask_i & ~union_others
            if lost == 0:
                # Redundant
                chosen_indices.pop(i)
                chosen_masks.pop(i)
                return chosen_indices, chosen_masks
            
            # Find a block that covers all lost
            for i_new in range(num_blocks):
                if i_new in chosen_indices:
                    continue
                m_new = block_masks[i_new]
                if (lost & ~m_new) == 0:
                    # Valid replacement
                    chosen_indices[i] = i_new
                    chosen_masks[i] = m_new
                    return chosen_indices, chosen_masks
        
        return chosen_indices, chosen_masks
    
    # Main search: multiple restarts with different strategies
    best_result = None
    best_count = float('inf')
    
    # Deterministic greedy first
    det_indices, det_masks = greedy_set_cover(randomize=False)
    det_indices, det_masks = remove_redundant(det_indices, det_masks)
    
    # Local search on deterministic result
    improved = True
    while improved:
        if time.time() > deadline:
            break
        old_count = len(det_indices)
        det_indices, det_masks = local_search_1_exchange(det_indices, det_masks)
        det_indices, det_masks = remove_redundant(det_indices, det_masks)
        if len(det_indices) < old_count:
            improved = True
        else:
            improved = False
    
    if len(det_indices) < best_count:
        best_count = len(det_indices)
        best_result = (list(det_indices), list(det_masks))
    
    # Randomized greedy with multiple restarts
    rng = random.Random(42)
    
    num_restarts = 10  # Will be limited by time
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary temperature and randomization
        temp = 0.5 + restart * 0.3
        rand_indices, rand_masks = greedy_set_cover(randomize=True, temperature=temp, rng=rng)
        
        if len(rand_indices) == 0:
            continue
        
        # Remove redundant
        rand_indices, rand_masks = remove_redundant(rand_indices, rand_masks)
        
        # Local search
        improved = True
        while improved:
            if time.time() > deadline:
                break
            old_count = len(rand_indices)
            rand_indices, rand_masks = local_search_1_exchange(rand_indices, rand_masks)
            rand_indices, rand_masks = remove_redundant(rand_indices, rand_masks)
            if len(rand_indices) < old_count:
                improved = True
            else:
                improved = False
        
        if len(rand_indices) < best_count:
            best_count = len(rand_indices)
            best_result = (list(rand_indices), list(rand_masks))
    
    # Additional: try to further reduce best result
    if best_result is not None:
        best_idx, best_msk = best_result
        # More aggressive local search
        for _ in range(20):
            if time.time() > deadline:
                break
            old_count = len(best_idx)
            best_idx, best_msk = local_search_1_exchange(best_idx, best_msk)
            best_idx, best_msk = remove_redundant(best_idx, best_msk)
            if len(best_idx) < old_count:
                best_count = len(best_idx)
            else:
                break
    
    if best_result is None:
        # Fallback: return empty or minimal
        return []
    
    return [list(all_blocks[idx]) for idx in best_result[0]]
