import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Precomputation
    # Generate all triples and map them to indices
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {trip: i for i, trip in enumerate(triples)}
    
    # Generate all possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute which triples each block covers
    # block_triple_masks[i] is a numpy array of triple indices covered by block i
    block_triple_masks = np.zeros((num_blocks, 1), dtype=object) # Using object to store variable length arrays? No, length is fixed C(k,t)
    mask_len = len(list(itertools.combinations(range(k), t)))
    block_triple_masks = np.zeros((num_blocks, mask_len), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        block_set = set(block)
        mask = []
        for trip in itertools.combinations(block, t):
            mask.append(triple_to_idx[trip])
        block_triple_masks[i] = mask
        
    # 2. Greedy Construction
    uncovered = np.ones(num_triples, dtype=bool)
    cover_count = np.zeros(num_triples, dtype=np.int32)
    blocks = []
    block_indices = []
    
    # Greedy: pick block covering most uncovered triples
    # To speed up, we can just iterate all blocks and find max gain.
    # num_blocks is ~350k. num_triples is ~2k.
    # 350k * 2k operations is 700 million, too slow in Python loop.
    # We need to be smarter or use numpy vectorization.
    
    # Let's use a different greedy strategy:
    # Maintain a list of uncovered triples.
    # For each candidate block, count how many of its triples are in uncovered.
    
    # Actually, for v=24, k=7, t=3, the number of blocks is manageable.
    # Let's try to optimize the greedy step.
    # We can compute the gain for all blocks at once?
    # gain[i] = sum(uncovered[j] for j in block_triple_masks[i])
    # This is a matrix multiplication or similar.
    # uncovered is a vector of 0/1.
    # We can create a sparse matrix M where M[i, j] = 1 if block i covers triple j.
    # Then gain = M @ uncovered_vector.
    
    # Building M: 346k x 2024 sparse matrix.
    from scipy import sparse
    
    rows = []
    cols = []
    for i in range(num_blocks):
        for j in block_triple_masks[i]:
            rows.append(i)
            cols.append(j)
            
    M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_blocks, num_triples))
    
    # Greedy Loop
    while np.any(uncovered):
        if time.time() > deadline - 5:
            break
            
        # Calculate gain for each block
        # gain = M @ uncovered (where uncovered is 0/1)
        gain = M @ uncovered.astype(np.int32)
        
        # Find block with max gain
        best_block_idx = np.argmax(gain)
        
        # If gain is 0, we are stuck (shouldn't happen if uncovered is not empty and M is valid)
        if gain[best_block_idx] == 0:
             break
            
        # Add block
        block_indices.append(best_block_idx)
        blocks.append(list(all_blocks[best_block_idx]))
        
        # Update cover_count and uncovered
        mask = block_triple_masks[best_block_idx]
        cover_count[mask] += 1
        uncovered[mask] = False
        
    # 3. Local Search (Reduction and Swap)
    # We have a valid cover. Now minimize size.
    
    def is_redundant(block_idx):
        mask = block_triple_masks[block_idx]
        # A block is redundant if all its triples have cover_count > 1
        # i.e., min(cover_count[mask]) > 1
        return np.min(cover_count[mask]) > 1
        
    def try_remove_redundant():
        removed_any = False
        # Iterate through current blocks
        # We need to be careful with indices if we remove
        new_block_indices = []
        for idx in block_indices:
            if is_redundant(idx):
                # Remove it
                mask = block_triple_masks[idx]
                cover_count[mask] -= 1
                # It becomes uncovered? No, if it was redundant, cover_count was >= 2, so now >= 1.
                # uncovered should remain False.
                removed_any = True
            else:
                new_block_indices.append(idx)
        
        if removed_any:
            block_indices[:] = new_block_indices
            # Update blocks list
            blocks[:] = [list(all_blocks[idx]) for idx in block_indices]
            return True
        return False
        
    def try_swap():
        # Pick a random block to replace
        if not block_indices:
            return False
            
        old_idx_pos = np.random.randint(0, len(block_indices))
        old_idx = block_indices[old_idx_pos]
        
        # Pick a candidate new block
        # Heuristic: Pick a block that covers many triples with low cover_count?
        # Or just random.
        # Let's try random first.
        new_idx = np.random.randint(0, num_blocks)
        
        if new_idx == old_idx:
            return False
            
        # Check if swap is valid
        # Condition: For all triples in old_idx, if cover_count[trip] == 1, then new_idx MUST cover trip.
        old_mask = block_triple_masks[old_idx]
        new_mask = block_triple_masks[new_idx]
        
        # Triples in old_mask that are NOT in new_mask must have cover_count > 1
        # Triples in new_mask that are NOT in old_mask can have any cover_count (will increase)
        
        # Efficient check:
        # 1. Identify triples in old_mask where cover_count == 1.
        # 2. These MUST be in new_mask.
        
        critical_triples = old_mask[cover_count[old_mask] == 1]
        
        if len(critical_triples) > 0:
            # Check if all critical_triples are in new_mask
            # new_mask is a sorted array? No, it's generated by itertools.combinations which is sorted.
            # We can use set intersection or search.
            # Since mask size is small (35), set is fast.
            new_mask_set = set(new_mask)
            for trip in critical_triples:
                if trip not in new_mask_set:
                    return False
                    
        # Swap is valid
        # Update cover_count
        cover_count[old_mask] -= 1
        cover_count[new_mask] += 1
        
        # Update block_indices
        block_indices[old_idx_pos] = new_idx
        blocks[old_idx_pos] = list(all_blocks[new_idx])
        
        return True

    # Main Local Search Loop
    last_improvement_time = time.time()
    
    while time.time() < deadline - 2:
        # 1. Try to remove redundant blocks
        if try_remove_redundant():
            last_improvement_time = time.time()
            continue
            
        # 2. If no reduction, try swaps
        # Do a few swaps per iteration to explore the space
        swaps_done = 0
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            if try_swap():
                swaps_done += 1
                # After a swap, a block might become redundant, so we break to re-evaluate redundancy
                break
                
    # Final cleanup: remove any remaining redundant blocks
    while try_remove_redundant():
        pass
        
    return blocks

