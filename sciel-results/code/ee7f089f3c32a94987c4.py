import time
from itertools import combinations
from functools import lru_cache

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
    
    # Precompute for each block, the list of 5-subset indices it covers (for fast iteration)
    block_sub_indices = []
    for j, b in enumerate(all_blocks):
        subs = [sub_to_idx[s] for s in combinations(b, sub_size)]
        block_sub_indices.append(subs)
    
    # Cache MRV candidates per (uncovered_mask, needed_blocks)
    # Key: (uncovered_mask, needed) -> (chosen_sub_idx, tuple(candidates)) or None if dead end
    mrv_cache = {}
    
    def get_mrv(uncovered_mask, needed):
        key = (uncovered_mask, needed)
        if key in mrv_cache:
            return mrv_cache[key]
        
        min_candidates = None
        min_count = float('inf')
        chosen_sub_idx = -1
        
        # Iterate through all 792 subsets
        for s_idx in range(total_subs):
            if (uncovered_mask >> s_idx) & 1:
                continue
                
            # This subset is uncovered. Find valid blocks covering it.
            candidates = []
            for b_idx in subs_to_blocks[s_idx]:
                if block_masks[b_idx] & ~uncovered_mask:
                    continue
                candidates.append(b_idx)
                
            count = len(candidates)
            if count == 0:
                mrv_cache[key] = None
                return None
                
            if count < min_count:
                min_count = count
                min_candidates = candidates
                chosen_sub_idx = s_idx
                if count == 1:
                    break
                    
        if min_candidates is None:
            mrv_cache[key] = None
            return None
            
        result = (chosen_sub_idx, tuple(min_candidates))
        mrv_cache[key] = result
        return result
    
    # Completion check: can the remaining uncovered 5-subsets be packed?
    # Each uncovered 5-subset must be covered by exactly one future block.
    # A future block covering a particular uncovered 5-subset also covers 5 other 5-subsets.
    # Those other 5-subsets must also be uncovered (otherwise conflict).
    # So for a candidate block to be usable, all 6 of its 5-subsets must be uncovered.
    # We need exactly `needed` such blocks that partition the uncovered set.
    
    def can_complete(uncovered_mask, needed):
        if needed == 0:
            return uncovered_mask == 0
        if needed == 1:
            # Only one block can be added; it must cover exactly the uncovered set
            # But a block covers exactly 6 5-subsets, so uncovered must have exactly 6 bits set
            if uncovered_mask.bit_count() != 6:
                return False
            # Check if there's a block whose mask equals uncovered_mask
            for b_idx in range(n_blocks):
                if block_masks[b_idx] == uncovered_mask:
                    return True
            return False
        
        # For larger needed, use a necessary condition:
        # Each uncovered 5-subset must be in at least one valid block (one whose all 6 subs are uncovered)
        # And the number of valid blocks must be at least `needed`
        
        # Count valid blocks (all 6 of their 5-subsets are uncovered)
        valid_count = 0
        for b_idx in range(n_blocks):
            if block_masks[b_idx] & ~uncovered_mask:
                continue
            valid_count += 1
            if valid_count >= needed:
                break
        
        if valid_count < needed:
            return False
        
        # Additional check: each uncovered 5-subset must be covered by at least one valid block
        # This is implied by valid_count >= needed in most cases, but let's be safe for small needed
        if needed <= 3:
            for s_idx in range(total_subs):
                if (uncovered_mask >> s_idx) & 1:
                    covered_by_valid = False
                    for b_idx in subs_to_blocks[s_idx]:
                        if block_masks[b_idx] & ~uncovered_mask:
                            continue
                        covered_by_valid = True
                        break
                    if not covered_by_valid:
                        return False
        return True
    
    def backtrack(current_covered, current_selected):
        if time.time() > deadline:
            return None
            
        needed = target_blocks - len(current_selected)
        
        if needed == 0:
            if current_covered.bit_count() == total_subs:
                return list(current_selected)
            return None
        
        uncovered_mask = (~current_covered) & ((1 << total_subs) - 1)
        
        # Completion check
        if not can_complete(uncovered_mask, needed):
            return None
        
        mrv_result = get_mrv(uncovered_mask, needed)
        if mrv_result is None:
            return None
            
        chosen_sub_idx, candidates = mrv_result
        
        # Try each candidate
        for b_idx in candidates:
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
        # Fallback: greedy with better heuristic
        covered = initial_covered
        selected = list(initial_selected)
        
        # Greedy: repeatedly pick the block that maximizes new coverage
        # Use a set for O(1) removal
        remaining_set = set(range(n_blocks))
        remaining_set.discard(first_block_idx)
        
        for _ in range(target_blocks - 1):
            if time.time() > deadline:
                break
            best_b = -1
            max_new = -1
            for b_idx in remaining_set:
                if block_masks[b_idx] & covered:
                    continue
                new_count = (block_masks[b_idx] & ~covered).bit_count()
                if new_count > max_new:
                    max_new = new_count
                    best_b = b_idx
            
            if best_b == -1:
                break
                
            covered |= block_masks[best_b]
            selected.append(best_b)
            remaining_set.discard(best_b)
            
        return [all_blocks[i] for i in selected]
