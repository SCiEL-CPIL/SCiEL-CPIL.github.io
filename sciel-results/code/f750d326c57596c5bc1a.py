import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using a bitmask-based exact cover solver
    with MRV heuristic and symmetry breaking.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    total_subs = 792 # C(12, 5)
    
    # 1. Precomputation
    all_points = range(n_points)
    all_subs = list(combinations(all_points, sub_size))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    all_blocks = list(combinations(all_points, block_size))
    n_blocks = len(all_blocks)
    
    # Map each block to a bitmask of the 5-subsets it covers
    # Python integers are arbitrary precision, so we can use them as bitsets
    block_masks = [0] * n_blocks
    for j, b in enumerate(all_blocks):
        mask = 0
        for s in combinations(b, sub_size):
            mask |= (1 << sub_to_idx[s])
        block_masks[j] = mask
        
    # Inverted index: for each 5-subset index, list of block indices containing it
    subs_to_blocks = [[] for _ in range(total_subs)]
    for j in range(n_blocks):
        # Iterate through the 6 subsets of the block
        b = all_blocks[j]
        for s in combinations(b, sub_size):
            idx = sub_to_idx[s]
            subs_to_blocks[idx].append(j)

    # Symmetry breaking: Fix the first block to be (0, 1, 2, 3, 4, 5)
    first_block_idx = None
    for j, b in enumerate(all_blocks):
        if b == (0, 1, 2, 3, 4, 5):
            first_block_idx = j
            break
            
    if first_block_idx is None:
        return [] # Should not happen
        
    initial_covered = block_masks[first_block_idx]
    initial_selected = [first_block_idx]
    
    # Backtracking search
    # We need to find 131 more blocks
    # State: current_covered_mask, current_selected_list
    
    def backtrack(current_covered, current_selected):
        if time.time() > deadline:
            return None
            
        if len(current_selected) == target_blocks:
            # Check if we covered all 792 subsets
            # Since we only add disjoint blocks, and we have 132 blocks * 6 subs/block = 792 subs,
            # if we haven't hit a conflict, we must have covered exactly 792 distinct subs.
            # We just need to verify the count of bits set is 792.
            if current_covered.bit_count() == total_subs:
                return list(current_selected)
            return None
            
        # Find the uncovered 5-subset with the fewest valid candidate blocks (MRV)
        min_candidates = None
        min_count = float('inf')
        chosen_sub_idx = -1
        
        # Iterate through all 792 subsets
        for s_idx in range(total_subs):
            # Check if this subset is already covered
            if (current_covered >> s_idx) & 1:
                continue
                
            # This subset is uncovered. Find valid blocks covering it.
            candidates = []
            for b_idx in subs_to_blocks[s_idx]:
                # Check if this block conflicts with current_covered
                if block_masks[b_idx] & current_covered:
                    continue
                # Also, we should avoid re-selecting the same block?
                # The problem asks for distinct 6-subsets.
                # Since blocks are disjoint in terms of 5-subsets, and 5-subsets determine the block?
                # No, two different 6-blocks can share a 5-subset? No, the constraint is "no 5-subset lies in two blocks".
                # So if two blocks share a 5-subset, they cannot both be in the solution.
                # Our conflict check `block_masks[b_idx] & current_covered` ensures that the new block
                # does not share ANY 5-subset with any previously selected block.
                # Therefore, all selected blocks are distinct and disjoint in terms of 5-subsets.
                candidates.append(b_idx)
                
            count = len(candidates)
            if count == 0:
                return None # Dead end
                
            if count < min_count:
                min_count = count
                min_candidates = candidates
                chosen_sub_idx = s_idx
                if count == 1:
                    break # Can't get better than 1
                    
        if min_candidates is None:
            return None
            
        # Try each candidate
        for b_idx in min_candidates:
            new_covered = current_covered | block_masks[b_idx]
            new_selected = current_selected + [b_idx]
            result = backtrack(new_covered, new_selected)
            if result is not None:
                return result
                
        return None

    solution_indices = backtrack(initial_covered, initial_selected)
    
    if solution_indices:
        # Convert indices back to tuples
        return [all_blocks[i] for i in solution_indices]
    else:
        # Fallback: Return an empty list or a partial solution if time runs out
        # The task says "Maximize the number of blocks".
        # If we fail to find 132, we should return the best we found?
        # The current program returns a greedy fallback.
        # However, the MRV search is very strong for this instance.
        # If it fails, it's likely due to time limit.
        # Let's return the initial block at least? Or empty?
        # The evaluator measures raw value. 0 blocks is bad.
        # Let's implement a simple greedy fallback just in case.
        
        covered = initial_covered
        selected = list(initial_selected)
        
        # Greedy: Pick blocks that cover the most new subsets
        # We need to pick 131 more.
        remaining_blocks = [i for i in range(n_blocks) if i != first_block_idx]
        
        # Sort by heuristic? Or just iterate?
        # A better greedy: repeatedly pick the block that adds the most new 5-subsets.
        
        for _ in range(target_blocks - 1):
            if time.time() > deadline:
                break
            best_b = -1
            max_new = -1
            for b_idx in remaining_blocks:
                if block_masks[b_idx] & covered:
                    continue
                new_mask = block_masks[b_idx] & ~covered
                new_count = new_mask.bit_count()
                if new_count > max_new:
                    max_new = new_count
                    best_b = b_idx
            
            if best_b == -1:
                break
                
            covered |= block_masks[best_b]
            selected.append(best_b)
            remaining_blocks.remove(best_b)
            
        return [all_blocks[i] for i in selected]
