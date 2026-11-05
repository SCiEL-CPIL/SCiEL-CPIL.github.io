"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using Local Search (Swap and Delete).
    """
    start_time = time.time()
    time_limit = 115  # Keep safe margin
    
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block to list of triple indices
    block_triples = []
    # Map triple index to list of block indices containing it
    triple_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        triples_in_block = []
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            triples_in_block.append(t_idx)
            triple_to_blocks[t_idx].append(b_idx)
        block_triples.append(triples_in_block)
        
    # 2. Initialization: Greedy Set Cover
    # Use bitsets for speed in initialization
    block_bitsets = []
    for block in all_blocks:
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        block_bitsets.append(mask)
        
    full_mask = (1 << num_triples) - 1
    uncovered_mask = full_mask
    selected_block_indices = []
    used_block_indices = set()
    
    # Greedy
    for _ in range(num_triples + 10):
        if uncovered_mask == 0:
            break
        best_block_idx = -1
        best_gain = 0
        for i, mask in enumerate(block_bitsets):
            if i in used_block_indices:
                continue
            gain = (uncovered_mask & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        if best_block_idx == -1:
            break
        selected_block_indices.append(best_block_idx)
        used_block_indices.add(best_block_idx)
        uncovered_mask &= ~block_bitsets[best_block_idx]
        
    current_blocks = [all_blocks[i] for i in selected_block_indices]
    current_block_set = set(selected_block_indices)
    
    # Helper: Check if a set of block indices forms a valid cover
    # We maintain coverage counts for efficiency
    def get_coverage_counts(block_indices):
        counts = np.zeros(num_triples, dtype=np.int32)
        for b_idx in block_indices:
            for t_idx in block_triples[b_idx]:
                counts[t_idx] += 1
        return counts
        
    # Helper: Find redundant blocks
    def find_redundant_blocks(block_indices, counts):
        redundant = []
        for b_idx in block_indices:
            is_redundant = True
            for t_idx in block_triples[b_idx]:
                if counts[t_idx] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant.append(b_idx)
        return redundant

    # Helper: Find swap candidates
    # Given a block b_out, find b_in not in current_set such that 
    # all unique triples of b_out are covered by b_in.
    def find_swap(b_out_idx, current_block_set, counts):
        # Find unique triples of b_out
        unique_triples = []
        for t_idx in block_triples[b_out_idx]:
            if counts[t_idx] == 1:
                unique_triples.append(t_idx)
                
        if not unique_triples:
            return None # Should be handled by deletion
            
        # Find intersection of blocks covering these unique triples
        # Start with blocks covering the first unique triple
        candidates = set(triple_to_blocks[unique_triples[0]])
        for t_idx in unique_triples[1:]:
            candidates &= set(triple_to_blocks[t_idx])
            
        # Remove current blocks
        candidates -= current_block_set
        
        if not candidates:
            return None
            
        return candidates

    # 3. Local Search Loop
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        
        # Get current coverage
        current_indices = list(current_block_set)
        counts = get_coverage_counts(current_indices)
        
        # Phase 1: Deletion
        redundant_blocks = find_redundant_blocks(current_indices, counts)
        if redundant_blocks:
            # Remove one (or all) redundant blocks
            # Removing all at once is safe if they are truly redundant based on current counts
            # But removing one might make another redundant or non-redundant? 
            # If A is redundant, all its triples have count > 1. Removing A decreases counts.
            # It doesn't make other blocks "less redundant" in terms of their own unique triples 
            # becoming unique? No, it might make a triple that was count 2 become count 1.
            # So we should remove one by one or re-evaluate.
            # Let's remove one at a time to be safe and simple.
            b_to_remove = random.choice(redundant_blocks)
            current_block_set.remove(b_to_remove)
            improved = True
            continue # Restart loop to re-calculate counts
            
        # Phase 2: Swap
        # Try to swap each block to see if it enables a deletion later
        # We iterate through blocks in random order
        blocks_to_try = list(current_block_set)
        random.shuffle(blocks_to_try)
        
        swap_made = False
        for b_out_idx in blocks_to_try:
            if time.time() - start_time > time_limit:
                break
                
            # Re-check if b_out_idx is still in set (it is, we didn't modify set in this phase yet)
            # Re-calculate unique triples for this specific block? 
            # We have `counts` from before deletion phase. Since no deletion happened, counts are valid.
            unique_triples = []
            for t_idx in block_triples[b_out_idx]:
                if counts[t_idx] == 1:
                    unique_triples.append(t_idx)
                    
            if not unique_triples:
                # This block is redundant, but Phase 1 didn't catch it? 
                # Maybe counts changed? No. 
                # Just remove it.
                current_block_set.remove(b_out_idx)
                improved = True
                swap_made = True
                break
                
            candidates = find_swap(b_out_idx, current_block_set, counts)
            if candidates:
                # Pick a random candidate
                b_in_idx = random.choice(list(candidates))
                
                # Perform swap
                current_block_set.remove(b_out_idx)
                current_block_set.add(b_in_idx)
                
                # After swap, check if we can delete anything immediately
                # Update counts
                # Remove b_out_idx contributions, add b_in_idx contributions
                for t_idx in block_triples[b_out_idx]:
                    counts[t_idx] -= 1
                for t_idx in block_triples[b_in_idx]:
                    counts[t_idx] += 1
                    
                # Check for redundancies with new counts
                new_redundant = find_redundant_blocks(list(current_block_set), counts)
                if new_redundant:
                    # We have a valid cover of same size, but now have redundancy.
                    # The next loop iteration (Phase 1) will delete it, reducing size.
                    # This is a good move.
                    improved = True
                    swap_made = True
                    break
                else:
                    # Swap maintained validity but no immediate reduction.
                    # We might still accept it to explore state space?
                    # Usually, we only accept if it leads to reduction or is "better" in some metric.
                    # For now, let's only accept if it leads to immediate reduction potential 
                    # or if we are stuck. 
                    # Actually, swapping without reduction might lead to a dead end.
                    # Let's revert this swap if it didn't lead to a deletion.
                    # Revert
                    current_block_set.remove(b_in_idx)
                    current_block_set.add(b_out_idx)
                    for t_idx in block_triples[b_in_idx]:
                        counts[t_idx] -= 1
                    for t_idx in block_triples[b_out_idx]:
                        counts[t_idx] += 1
                    # Continue trying other swaps
                    
        if not swap_made:
            # No swap led to immediate improvement.
            # We might be stuck in a local minimum.
            # To escape, we could accept a swap that doesn't reduce size but changes the structure,
            # but that risks increasing size if we later fail to delete.
            # A safer strategy: If stuck, break.
            # Or, try a "random swap" and hope for the best?
            # Given time constraints, breaking is safer to ensure validity.
            # However, simple hill climbing often gets stuck.
            # Let's try one more phase: Random Restart of the search order?
            # Or just break.
            break

    return [all_blocks[i] for i in current_block_set]

# EVOLVE-BLOCK-END
