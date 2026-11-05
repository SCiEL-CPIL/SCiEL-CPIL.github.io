import itertools
import random
import time
import numpy as np

# Global Precomputation for v=19, k=7, t=4
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets) # 3876

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
# For each subset, list of blocks containing it
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_len = float('inf')
    best_blocks = []
    
    # We will run multiple restarts with different random seeds
    # Each restart consists of:
    # 1. Greedy Initialization (Constrained Subsets)
    # 2. Local Search (3-step moves: Remove B_out, Add B_in, Remove B_redundant)
    # 3. Perturbation if stuck
    
    num_restarts = 10 # Fewer, deeper restarts
    
    for restart in range(num_restarts):
        random.seed(42 + restart * 100)
        
        current_blocks = []
        counts = [0] * N_SUBSETS
        block_in_use = [False] * N_BLOCKS
        
        def add_block(idx):
            current_blocks.append(idx)
            block_in_use[idx] = True
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                mm ^= lsb

        def remove_block(i):
            idx = current_blocks[i]
            block_in_use[idx] = False
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] -= 1
                mm ^= lsb
            current_blocks.pop(i)

        def is_redundant(i):
            m = all_masks[current_blocks[i]]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        # 1. Greedy Initialization: Pick most constrained uncovered subset
        # To avoid O(N_SUBSETS * N_BLOCKS) scan, we maintain a list of uncovered subsets
        # and pick one with the fewest covering blocks (or just the first uncovered).
        # "Most constrained" is better.
        uncovered_mask = (1 << N_SUBSETS) - 1
        # Precompute number of blocks containing each subset for tie-breaking? 
        # blocks_containing_subset[idx] length is constant C(15,3)=455. 
        # So all subsets are equally constrained initially.
        # We just pick the first uncovered subset.
        
        while uncovered_mask:
            # Find first uncovered subset
            # To speed up, we can iterate through bits
            mm = uncovered_mask
            if not mm:
                break
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            uncovered_mask ^= lsb # Mark as "to be covered" temporarily? No, we cover it.
            
            # Find best block covering bit_idx
            # Best = covers most OTHER uncovered subsets
            candidates = blocks_containing_subset[bit_idx]
            best_block = -1
            best_gain = -1
            
            # To make it diverse, we can sample candidates or pick random from top
            # For now, pick the one with max gain
            for b_idx in candidates:
                if block_in_use[b_idx]:
                    continue
                # Calculate gain: number of bits in all_masks[b_idx] that are in uncovered_mask
                # Note: uncovered_mask still contains bit_idx, so gain >= 1
                gain = (all_masks[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    # Early termination if perfect gain (35)
                    if gain == 35:
                        break
            
            if best_block != -1:
                add_block(best_block)
                # Update uncovered_mask
                uncovered_mask &= ~all_masks[best_block]
            else:
                # Should not happen
                break
        
        # Initial Pruning
        i = 0
        while i < len(current_blocks):
            if is_redundant(i):
                remove_block(i)
            else:
                i += 1
                
        current_len = len(current_blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = current_blocks.copy()
            
        # 2. Local Search: 3-step moves
        # Move: Remove B_out, Add B_in, Remove B_redundant
        # Net change: -1 block
        
        improved = True
        local_deadline = start_time + (deadline - start_time) / (num_restarts * 2)
        
        while improved and time.time() < local_deadline:
            improved = False
            # Iterate through blocks in random order to find a 3-step move
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > local_deadline:
                    break
                if i >= len(current_blocks):
                    continue
                    
                b_out_idx = current_blocks[i]
                m_out = all_masks[b_out_idx]
                
                # Find subsets covered ONLY by b_out
                unique_bits = []
                mm = m_out
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 1:
                        unique_bits.append(bit_idx)
                    mm ^= lsb
                
                if not unique_bits:
                    # It's redundant, remove it
                    remove_block(i)
                    improved = True
                    # Restart loop as indices changed
                    break
                
                # Find candidates B_in that cover all unique_bits
                # Start with candidates for the first unique bit
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                # Filter out b_out and already used blocks
                valid_candidates = [c for c in candidates if c != b_out_idx and not block_in_use[c]]
                
                if not valid_candidates:
                    continue
                
                # Try a few random candidates (to avoid getting stuck in local opt of B_in choice)
                random.shuffle(valid_candidates)
                for b_in_idx in valid_candidates[:5]: # Limit checks
                    # Simulate: Remove b_out, Add b_in
                    # We need to check if any block becomes redundant
                    
                    # 1. Temporarily remove b_out
                    # Instead of actually removing/adding, we can calculate new counts
                    # new_counts = counts - mask(b_out) + mask(b_in)
                    # Then check if any block j (j != i, j != b_in_idx) has all bits with count >= 2 in new_counts?
                    # Actually, a block j is redundant if all its bits have count >= 2 in new_counts.
                    
                    m_in = all_masks[b_in_idx]
                    
                    # Check for redundant blocks after swap
                    # A block j becomes redundant if for all bits in m_j, new_count[bit] >= 2.
                    # new_count[bit] = counts[bit] - (1 if bit in m_out else 0) + (1 if bit in m_in else 0)
                    
                    # This check is O(N_BLOCKS * 35). Too slow for every candidate.
                    # Optimization: Only check blocks that share bits with b_out or b_in.
                    # Blocks that don't share bits with b_out or b_in have unchanged counts.
                    # If they were not redundant before, they won't be now.
                    # If they were redundant before, they would have been removed in the previous pruning.
                    # So we only need to check blocks that intersect with (m_out U m_in).
                    
                    # Identify candidate blocks for redundancy check
                    # These are blocks in current_blocks (except i) that intersect with m_out or m_in
                    # And potentially b_in_idx itself? No, b_in is new.
                    
                    # Let's just do the actual swap and prune, it's simpler and faster in Python if we optimize the prune.
                    # Swap:
                    remove_block(i)
                    add_block(b_in_idx)
                    
                    # Prune: Find any redundant block
                    redundant_found = -1
                    for j in range(len(current_blocks)):
                        if j == len(current_blocks) - 1: # b_in_idx is at the end
                             # Check if b_in is redundant? Unlikely, it covers unique bits of b_out.
                             # But if b_in covers bits covered by others, it might be.
                             pass
                        if is_redundant(j):
                            redundant_found = j
                            break
                    
                    if redundant_found != -1:
                        # Success! Remove the redundant block
                        remove_block(redundant_found)
                        improved = True
                        # Keep the swap and removal
                        # Break out to restart the scan with new state
                        break
                    else:
                        # Failure, revert swap
                        # Remove b_in_idx (which is at the end)
                        remove_block(len(current_blocks) - 1)
                        # Re-add b_out_idx
                        add_block(b_out_idx)
                        # Note: Order of blocks in current_blocks might change, but logic holds.
                        # However, 'i' index is no longer valid for b_out_idx.
                        # We just break out of candidate loop and continue to next 'i' in outer loop?
                        # Actually, we reverted, so state is same.
                        # We can break candidate loop to try next i.
                        break
                
                if improved:
                    break
            
            if improved:
                # Update best
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
            
            # If no improvement, perturb
            if not improved and time.time() < local_deadline:
                # Random perturbation: Swap a random block with a random other block
                if len(current_blocks) > 1:
                    i = random.randint(0, len(current_blocks) - 1)
                    b_out_idx = current_blocks[i]
                    # Find a random block not in use
                    b_in_idx = random.randint(0, N_BLOCKS - 1)
                    while block_in_use[b_in_idx] or b_in_idx == b_out_idx:
                        b_in_idx = random.randint(0, N_BLOCKS - 1)
                    
                    # Perform swap blindly
                    remove_block(i)
                    add_block(b_in_idx)
                    
                    # Prune
                    j = 0
                    while j < len(current_blocks):
                        if is_redundant(j):
                            remove_block(j)
                        else:
                            j += 1
                            
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()

    # Final cleanup
    if not best_blocks:
        # Fallback if something went wrong
        return []
        
    return [list(all_blocks[i]) for i in best_blocks]
