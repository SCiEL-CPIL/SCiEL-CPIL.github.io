import itertools
import time
import random

# Precompute necessary data structures for the specific problem size if possible,
# but the function construct(v, k, t) must be general.
# However, the task specifies "the evaluator passes this task's parameters v, k and t".
# The parameters are fixed for the task instance (19, 8, 5).
# We can assume v=19, k=8, t=5 for optimization, but should handle general case gracefully.
# Given the strict time limit and specific task, we will optimize for the specific case
# while keeping the interface.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # If v is small, we can use bitsets.
    # Number of t-subsets
    num_t_subsets = 1
    for i in range(t):
        num_t_subsets = num_t_subsets * (v - i) // (i + 1)
        
    # If num_t_subsets is too large for efficient bitset ops in pure Python (though big ints are ok),
    # we might need a different strategy. But for 11628 bits, big ints are fine.
    
    # Map t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute mask for each block: which t-subsets it covers
    # mask[i] is a bitset (int) where bit j is 1 if t_subsets[j] is in block i
    block_masks = []
    for block in k_subsets:
        mask = 0
        # Efficient way: iterate over t-subsets of the block
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Initial full mask
    full_mask = (1 << num_t_subsets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Helper to count bits
    def popcount(x):
        return x.bit_count()
        
    # Run multiple greedy restarts
    while time.time() < deadline:
        uncovered = full_mask
        current_blocks = []
        used_blocks = [False] * len(k_subsets)
        
        while uncovered:
            # Find block with max gain
            max_gain = 0
            candidates = []
            # Optimization: If we have many blocks, iterating all is O(N).
            # N = 75582. This is acceptable.
            for i, mask in enumerate(block_masks):
                if used_blocks[i]:
                    continue
                gain = popcount(uncovered & mask)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered != 0
                
            # Pick one candidate
            # Random tie-breaking to explore different solutions
            chosen_idx = random.choice(candidates)
            
            current_blocks.append(chosen_idx)
            used_blocks[chosen_idx] = True
            uncovered &= ~block_masks[chosen_idx]
            
        # Local Search: Try to remove redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks.
        # We can check this by maintaining a coverage count for each t-subset.
        # But recomputing coverage counts from scratch is O(N_blocks * C(k,t)).
        # For 350 blocks * 56 t-subsets = ~20000 ops. Very fast.
        
        def try_reduce(blocks):
            # Return a reduced list of blocks
            changed = True
            while changed:
                changed = False
                # Count coverage for each t-subset
                coverage = [0] * num_t_subsets
                for b_idx in blocks:
                    mask = block_masks[b_idx]
                    # Iterate set bits in mask
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        coverage[idx] += 1
                        m ^= lsb
                        
                # Try removing each block
                new_blocks = []
                for b_idx in blocks:
                    mask = block_masks[b_idx]
                    # Check if any t-subset in this block has coverage == 1
                    # If so, it's critical, keep it.
                    # If all have coverage > 1, it's redundant, remove it.
                    is_critical = False
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if coverage[idx] == 1:
                            is_critical = True
                            break
                        m ^= lsb
                    
                    if is_critical:
                        new_blocks.append(b_idx)
                    else:
                        changed = True
                        # Update coverage for removed block
                        m = mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            coverage[idx] -= 1
                            m ^= lsb
                            
                blocks = new_blocks
            return blocks
        
        reduced_blocks = try_reduce(current_blocks)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        # Check time
        if time.time() > deadline - 5:
            break

    if best_blocks is None:
        # Fallback: return empty or seed-like result if no time passed
        # But we should have at least one iteration.
        return []
        
    # Convert indices back to lists of points
    final_blocks = [list(k_subsets[i]) for i in best_blocks]
    # Sort blocks for consistency
    final_blocks.sort()
    return final_blocks
