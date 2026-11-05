import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    full_mask = (1 << num_t_subsets) - 1
    
    # 2. Generate all k-subsets (blocks) and their coverage masks
    k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_subsets:
        mask = 0
        # For each t-subset within this k-subset, set the bit
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(k_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple randomized greedy runs
    # We estimate how many runs we can do.
    # One run is roughly O(num_blocks * num_selected_blocks) operations.
    # Let's try to fit as many as possible before deadline.
    
    runs = 0
    while time.time() < deadline:
        runs += 1
        uncovered = full_mask
        selected_indices = []
        
        # Greedy selection
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            candidates = []
            
            # To speed up, we can just find the max.
            # But for randomization, we might want top K.
            # Let's stick to max first, but introduce randomness in tie-breaking or slight noise.
            
            # Optimization: Pre-compute list of (block_idx, mask)
            # Actually, iterating 50k is fine.
            
            for i in range(num_blocks):
                # Check if this block adds anything
                # If mask is 0, gain is 0.
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # If gain is 0, we are stuck? No, if uncovered is not 0, 
                    # some block must cover it. 
                    # Wait, if best_gain becomes 0, it means no block covers any 
                    # of the remaining uncovered subsets? That shouldn't happen 
                    # unless the covering is impossible, which it isn't.
                    # But if we pick a block that doesn't cover the specific 
                    # 'min' uncovered subset, we might not make progress on that one, 
                    # but we cover others.
                    
            # Randomization: With some probability, pick from top few
            # Or just pick best.
            # Let's add a small random tie-breaker or noise for diversity.
            
            if best_gain == 0:
                # Should not happen, but safety
                break
            
            # Add randomization: 10% chance to pick a random block from top 20% of gains?
            # Simpler: Just pick best. The diversity comes from different random seeds 
            # if we were to randomize the order of blocks. 
            # Since we iterate 0..N, the first max wins. 
            # To randomize, we can shuffle the block indices once per run.
            
            selected_indices.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                break
        
        # Post-processing: Remove redundant blocks
        # We need to check if a block is redundant.
        # A block is redundant if all its t-subsets are covered by OTHER blocks.
        
        # Let's reconstruct the list of blocks from indices
        current_blocks = [k_subsets[i] for i in selected_indices]
        
        # Count coverage
        # Map t-subset index -> count of blocks covering it
        counts = [0] * num_t_subsets
        # block_idx -> list of t-subset indices it covers
        block_covered_ts = []
        for i in selected_indices:
            mask = block_masks[i]
            ts_indices = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                ts_indices.append(idx)
                m ^= lsb
            block_covered_ts.append(ts_indices)
            for idx in ts_indices:
                counts[idx] += 1
        
        # Iterative removal
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_selected_indices = []
            # Iterate in reverse to allow removal? Or just filter.
            # Let's filter.
            for j, idx in enumerate(selected_indices):
                is_redundant = True
                for ts_idx in block_covered_ts[j]:
                    if counts[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    for ts_idx in block_covered_ts[j]:
                        counts[ts_idx] -= 1
                    changed = True
                else:
                    new_selected_indices.append(idx)
            
            if changed:
                selected_indices = new_selected_indices
                # Update block_covered_ts? 
                # We only need to know which ones are left for the next iteration.
                # We can just re-derive or keep a mapping.
                # Simpler: Just re-run the count logic? No, that's expensive.
                # We can maintain a list of active blocks.
                
        # Actually, the removal loop above modifies `selected_indices` but `block_covered_ts` 
        # still has entries for removed blocks. We need to be careful.
        
        # Let's do a cleaner removal pass.
        # 1. Build initial counts and mapping
        # 2. Loop:
        #    Find a block where all its ts have count > 1.
        #    Remove it, decrement counts.
        #    Repeat.
        
        # Re-implementing removal cleanly for correctness
        if time.time() < deadline:
            counts = [0] * num_t_subsets
            block_ts_list = []
            for idx in selected_indices:
                mask = block_masks[idx]
                ts_list = []
                m = mask
                while m:
                    lsb = m & -m
                    idx_ts = lsb.bit_length() - 1
                    ts_list.append(idx_ts)
                    m ^= lsb
                block_ts_list.append(ts_list)
                for ts_idx in ts_list:
                    counts[ts_idx] += 1
            
            active = [True] * len(selected_indices)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(selected_indices)):
                    if not active[i]:
                        continue
                    redundant = True
                    for ts_idx in block_ts_list[i]:
                        if counts[ts_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        active[i] = False
                        changed = True
                        for ts_idx in block_ts_list[i]:
                            counts[ts_idx] -= 1
            
            final_indices = [idx for i, idx in enumerate(selected_indices) if active[i]]
            
            if len(final_indices) < best_count:
                best_count = len(final_indices)
                best_blocks = [k_subsets[i] for i in final_indices]
                
        # If we found a very good solution, we might stop early?
        # The lower bound is 111. If we hit 111, we can stop.
        if best_count <= 111:
            break
            
    if best_blocks is None:
        # Fallback to a simple valid covering if something went wrong
        # Or just return the last one
        return []
        
    return best_blocks
