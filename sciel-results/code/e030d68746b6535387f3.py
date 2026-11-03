import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    # A block is a k-subset. It covers all t-subsets within it.
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    
    # Precomputation cost: O(n_blocks * C(k,t))
    # 75582 * C(8,5) = 75582 * 56 = ~4.2M operations. This is fast.
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    def greedy_with_order(block_order):
        """Run greedy set cover with blocks considered in the given order for tie-breaking."""
        uncovered_mask = (1 << n_t) - 1
        selected_blocks = []
        removed = set()
        
        # Precompute full coverage count for each block (for initial sorting hint)
        # Not strictly needed but helps
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_block_idx = -1
            best_gain = 0
            
            # Iterate through blocks in the given order
            for i in block_order:
                if i in removed:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain == 0:
                    removed.add(i)
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    
            if best_block_idx == -1:
                # Should not happen if uncovered_mask is not 0
                break
            
            # Select the block
            selected_blocks.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected_blocks, uncovered_mask
    
    def remove_redundant(selected_blocks, block_masks, n_t):
        """Remove redundant blocks from the selected set."""
        if not selected_blocks:
            return selected_blocks
        
        # Initialize counts
        counts = [0] * n_t
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        # Iterative removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected = []
            for b_idx in selected_blocks:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove this block
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_selected.append(b_idx)
                    
            selected_blocks = new_selected
        
        return selected_blocks
    
    # Try multiple random orderings and keep the best result
    best_selected = None
    best_count = float('inf')
    
    # Number of restarts based on remaining time
    # Each greedy run takes roughly: n_iterations * (n_blocks / 2) average scans
    # Let's try a few restarts
    n_restarts = 3
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create a random order for tie-breaking
        if restart == 0:
            # First run: natural order
            block_order = list(range(n_blocks))
        else:
            block_order = list(range(n_blocks))
            random.shuffle(block_order)
        
        selected, remaining = greedy_with_order(block_order)
        
        if remaining != 0:
            # Incomplete coverage, skip
            continue
        
        # Remove redundant blocks
        selected = remove_redundant(selected, block_masks, n_t)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_selected = selected
        
        # If we've achieved a good solution, we can stop early
        if best_count <= 300:
            break
    
    if best_selected is None:
        # Fallback: use the first greedy result
        block_order = list(range(n_blocks))
        selected, remaining = greedy_with_order(block_order)
        best_selected = selected
    
    # Convert indices back to blocks
    return [list(all_blocks[i]) for i in best_selected]
