import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115
    
    # 1. Precomputation
    # Map t-subsets to indices
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Full mask: all bits 0 to n_tsubsets-1 are set
    full_mask = (1 << n_tsubsets) - 1
    
    # Precompute mask for each k-block
    all_blocks = list(itertools.combinations(all_points, k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            mask |= (1 << idx)
        block_masks[b] = mask
        
    # List of (block, mask) for faster access
    block_list = [(b, block_masks[b]) for b in all_blocks]
    
    # 2. Greedy Initialization
    # uncovered_mask starts as full_mask (all bits 1, meaning all are uncovered)
    # We want to cover them, so we want to clear bits.
    # Let's track `covered_mask`. Start at 0.
    # Gain of a block is popcount(covered_mask ^ (covered_mask | block_mask)) = popcount(block_mask & ~covered_mask)
    
    def greedy_init():
        covered = 0
        selected = []
        selected_set = set()
        
        # To speed up greedy, we can just iterate all blocks each time.
        # C(19,7) ~ 50k. Steps ~ 150. 50k * 150 = 7.5M operations.
        # Bitwise AND and popcount (bit_count) are very fast in Python 3.8+.
        
        while covered != full_mask:
            best_block = None
            best_gain = -1
            
            for b, mask in block_list:
                if b in selected_set:
                    continue
                # Calculate new bits covered
                new_bits = mask & ~covered
                gain = new_bits.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = (b, mask)
                    
            if best_block is None or best_gain == 0:
                break # Should not happen if parameters are valid
                
            selected.append(best_block)
            selected_set.add(best_block[0])
            covered |= best_block[1]
            
        return selected, covered

    # 3. Local Search: Reduce and Swap
    def reduce_and_swap(initial_selected, initial_covered):
        current_selected = list(initial_selected)
        current_covered = initial_covered
        best_selected = current_selected[:]
        best_covered = current_covered
        
        # Track best size
        best_size = len(current_selected)
        
        max_iterations = 20000
        current_time = time.time()
        
        # Helper to check if a block is redundant
        # Block (b, mask) is redundant if (current_covered ^ mask) & mask == 0
        # i.e., all bits in mask are already covered by OTHER blocks.
        # Since current_covered includes mask, current_covered ^ mask removes the contribution of this block.
        # If (current_covered ^ mask) & mask == mask, then mask is fully covered by others?
        # Wait.
        # Let C be the union of all masks.
        # Let M be the mask of block i.
        # C = M | (C \ M).
        # M is redundant if M is a subset of (C \ M).
        # This is equivalent to: (C \ M) & M == M.
        # C \ M is `current_covered ^ M` ONLY IF M is a subset of C (which it is).
        # So check: ((current_covered ^ mask) & mask) == mask
        
        def is_redundant(mask):
            return ((current_covered ^ mask) & mask) == mask

        for _ in range(max_iterations):
            if (time.time() - start_time) > time_limit:
                break
                
            # Phase 1: Try to remove redundant blocks
            removed_any = False
            # Iterate in reverse to allow safe popping
            for i in range(len(current_selected) - 1, -1, -1):
                b, mask = current_selected[i]
                if is_redundant(mask):
                    # Remove it
                    current_covered ^= mask
                    current_selected.pop(i)
                    removed_any = True
                    # Update best if we are still valid (which we are, by definition of redundancy)
                    if len(current_selected) < best_size:
                        best_size = len(current_selected)
                        best_selected = current_selected[:]
                        best_covered = current_covered
                    break # Restart reduction loop after a removal to check for new redundancies
            
            if not removed_any:
                # Phase 2: No redundant blocks found. Try a swap.
                if len(current_selected) > 0:
                    idx = random.randint(0, len(current_selected) - 1)
                    old_b, old_mask = current_selected[idx]
                    
                    # Pick a random new block not in current set
                    # To avoid infinite loop if set is dense, we can just pick random from all_blocks
                    # and check membership.
                    new_b, new_mask = random.choice(block_list)
                    if new_b in [x[0] for x in current_selected]:
                        continue
                    
                    # Apply swap
                    # New covered = (current_covered ^ old_mask) | new_mask
                    temp_covered = (current_covered ^ old_mask) | new_mask
                    
                    # Check if still valid (covers everything)
                    if temp_covered == full_mask:
                        # Accept swap
                        current_covered = temp_covered
                        current_selected[idx] = (new_b, new_mask)
                        
                        # After swap, we might have created a redundancy.
                        # We don't update best here because size is same.
                        # The next iteration's Phase 1 will check for removals.
                    else:
                        # Swap invalid, revert (do nothing)
                        pass
                else:
                    break # No blocks left, shouldn't happen for valid cover
                    
        return best_selected

    # Run
    init_selected, init_covered = greedy_init()
    result_selected = reduce_and_swap(init_selected, init_covered)
    
    # Extract blocks
    return [list(b) for b, mask in result_selected]
