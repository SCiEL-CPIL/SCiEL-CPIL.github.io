import time
import random
import sys
from itertools import combinations

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0  # 120 seconds limit
    
    # 1. Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(combinations(range(v), t))
    num_t = len(all_t_subsets)
    t_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # 2. Pre-compute all possible blocks and their covered t-subsets
    all_blocks = list(combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the list of t-subset indices it covers
    # This is a list of lists
    block_covers = []
    for b in all_blocks:
        covers = []
        for ts in combinations(b, t):
            covers.append(t_to_idx[ts])
        block_covers.append(covers)
        
    # Helper to check time
    def time_left():
        return deadline - (time.time() - start_time)
        
    # 3. Greedy Initialization
    # counts[i] = number of blocks covering t-subset i
    counts = [0] * num_t
    current_blocks = []
    block_indices = set() # Indices into all_blocks
    
    # To speed up greedy, we can just iterate
    # Find uncovered t-subsets
    uncovered = set(range(num_t))
    
    while uncovered and time_left() > 5.0:
        # Find block that covers most uncovered t-subsets
        best_block_idx = -1
        best_gain = -1
        # Randomize or use all? For init, checking all is fine if num_blocks is small.
        # num_blocks = 75582. This loop is 75k * 56 ops = 4M ops. Fast.
        for b_idx in range(num_blocks):
            covers = block_covers[b_idx]
            gain = 0
            for ts_idx in covers:
                if ts_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            # Early break if we found a perfect match? No, just max.
            
        if best_gain == 0:
            break # No block covers any uncovered? Should not happen if covering exists.
            
        current_blocks.append(best_block_idx)
        block_indices.add(best_block_idx)
        covers = block_covers[best_block_idx]
        for ts_idx in covers:
            counts[ts_idx] += 1
            if counts[ts_idx] == 1:
                uncovered.discard(ts_idx)
                
    # 4. Prune Redundant Blocks
    def prune():
        # Remove blocks that are redundant
        # A block is redundant if all its t-subsets are covered by other blocks
        # i.e., for all ts in block, counts[ts] > 1
        new_blocks = []
        for b_idx in current_blocks:
            is_redundant = True
            for ts_idx in block_covers[b_idx]:
                if counts[ts_idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(b_idx)
            else:
                # Remove it
                for ts_idx in block_covers[b_idx]:
                    counts[ts_idx] -= 1
                block_indices.discard(b_idx)
        return new_blocks

    current_blocks = prune()

    # 5. Simulated Annealing / Local Search
    # Objective: Minimize len(current_blocks)
    # Moves:
    # 1. Try to remove a block (if redundant)
    # 2. Try to swap a block with another to enable removal or just explore
    
    best_blocks = list(current_blocks)
    best_len = len(best_blocks)
    
    temperature = 1.0
    cooling = 0.999
    
    # We need a fast way to check if a swap is valid
    # Swap b1 (in) with b2 (out):
    # Valid if for all ts in b1, if counts[ts] == 1, then ts is in b2.
    
    while time_left() > 2.0:
        # Choose a move
        # Probability of remove: 0.3, swap: 0.7
        if random.random() < 0.3:
            # Try to remove a random block
            if not current_blocks:
                continue
            b1_idx_pos = random.randint(0, len(current_blocks) - 1)
            b1_idx = current_blocks[b1_idx_pos]
            
            # Check if redundant
            redundant = True
            for ts_idx in block_covers[b1_idx]:
                if counts[ts_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove
                for ts_idx in block_covers[b1_idx]:
                    counts[ts_idx] -= 1
                current_blocks.pop(b1_idx_pos)
                block_indices.discard(b1_idx)
                
                # Prune again? Maybe later.
                if len(current_blocks) < best_len:
                    best_blocks = list(current_blocks)
                    best_len = len(current_blocks)
                    temperature = 1.0 # Reset temp on improvement
            else:
                # Reject, or accept with penalty? 
                # Since we want to maintain validity, we reject.
                pass
        else:
            # Try to swap
            if not current_blocks:
                continue
            b1_idx_pos = random.randint(0, len(current_blocks) - 1)
            b1_idx = current_blocks[b1_idx_pos]
            
            # Choose b2 from all blocks not in current set
            # To be efficient, pick random b2
            b2_idx = random.randint(0, num_blocks - 1)
            if b2_idx in block_indices:
                continue
            
            # Check validity of swap
            # For all ts in b1:
            #   if counts[ts] == 1:
            #       b2 MUST cover ts
            valid = True
            covers_b1 = block_covers[b1_idx]
            covers_b2 = block_covers[b2_idx]
            
            # Optimization: Pre-compute a set for b2? Or just check membership?
            # covers_b2 is a list. Membership check in list is O(N).
            # N=56. So O(56*56) = 3136 ops. Fast enough.
            # But we can make a set for b2 on the fly? 
            # Or better, pre-compute a bitmask or boolean array for each block?
            # 75k blocks * 11k t-subsets is too big for bitmask in memory if using Python ints?
            # Python ints are arbitrary precision. 11628 bits is ~1.5KB per block. 75k * 1.5KB = 112MB.
            # This is feasible!
            
            # Let's re-initialize with bitmasks for speed?
            # No, let's stick to lists for now, but optimize the check.
            
            # Actually, let's just check the condition.
            for ts_idx in covers_b1:
                if counts[ts_idx] == 1:
                    # Check if ts_idx is in covers_b2
                    if ts_idx not in covers_b2:
                        valid = False
                        break
            
            if valid:
                # Perform swap
                for ts_idx in covers_b1:
                    counts[ts_idx] -= 1
                for ts_idx in covers_b2:
                    counts[ts_idx] += 1
                
                current_blocks[b1_idx_pos] = b2_idx
                block_indices.discard(b1_idx)
                block_indices.add(b2_idx)
                
                # Try to prune after swap
                # Find if any block is now redundant
                # This is O(N * 56). N~300. 16k ops.
                # We can do a quick prune.
                new_blocks = []
                for b_idx in current_blocks:
                    is_red = True
                    for ts_idx in block_covers[b_idx]:
                        if counts[ts_idx] == 1:
                            is_red = False
                            break
                    if not is_red:
                        new_blocks.append(b_idx)
                    else:
                        for ts_idx in block_covers[b_idx]:
                            counts[ts_idx] -= 1
                        block_indices.discard(b_idx)
                
                current_blocks = new_blocks
                
                if len(current_blocks) < best_len:
                    best_blocks = list(current_blocks)
                    best_len = len(current_blocks)
                    temperature = 1.0 # Reset
            
        temperature *= cooling
        
    # Convert best_blocks indices back to lists of points
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
