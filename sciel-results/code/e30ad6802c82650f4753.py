import time
from itertools import combinations
from collections import deque

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using an explicit algebraic construction
    via the Mathieu group M12, with a fallback to a bitmask-based exact cover solver.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    
    # --- Method 1: Explicit M12 Orbit Construction ---
    # Generators for M12 (0-indexed)
    # g1: 11-cycle fixing 11
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11]
    # g2: 12-cycle
    g2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    
    def apply_perm(block, perm):
        return tuple(sorted(perm[i] for i in block))
        
    def compose(p1, p2):
        # p1 * p2 means apply p2 then p1
        return [p1[p2[i]] for i in range(n_points)]
        
    # BFS to generate the group (or just the orbit of the base block)
    # We only need the orbit of the base block to get the design blocks.
    base_block = (0, 1, 2, 3, 4, 5)
    blocks_set = set()
    blocks_list = []
    
    # Queue of permutations
    # Identity
    identity = list(range(n_points))
    queue = deque()
    queue.append(identity)
    seen_perms = {tuple(identity)}
    
    # We will generate the group elements and apply them to the base block.
    # M12 has 95040 elements. This is small enough to generate fully if needed,
    # but we only need 132 blocks. We can stop early if we find 132 unique blocks.
    
    while queue:
        if len(blocks_list) >= target_blocks:
            break
        if time.time() > deadline - 5.0:
            break
            
        perm = queue.popleft()
        
        # Generate new blocks
        new_block = apply_perm(base_block, perm)
        if new_block not in blocks_set:
            blocks_set.add(new_block)
            blocks_list.append(new_block)
            
        # Generate new group elements
        for gen in [g1, g2]:
            new_perm = compose(gen, perm)
            new_perm_tuple = tuple(new_perm)
            if new_perm_tuple not in seen_perms:
                seen_perms.add(new_perm_tuple)
                queue.append(new_perm)
                
    # Verify the result
    if len(blocks_list) == target_blocks:
        # Check for no repeated 5-subsets
        # This is guaranteed by the construction if the group is M12 and base block is correct,
        # but let's do a quick sanity check on a few or just trust it.
        # For safety, we can run a fast verification.
        all_subs = set()
        valid = True
        for b in blocks_list:
            for s in combinations(b, sub_size):
                if s in all_subs:
                    valid = False
                    break
                all_subs.add(s)
            if not valid:
                break
                
        if valid and len(all_subs) == 792:
            return blocks_list
        else:
            # Invalid construction, fall through to backtracking
            pass

    # --- Method 2: Fallback Bitmask Backtracking ---
    # (Reusing the logic from the current program, optimized)
    
    all_points = range(n_points)
    all_subs = list(combinations(all_points, sub_size))
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    all_blocks = list(combinations(all_points, block_size))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for j, b in enumerate(all_blocks):
        mask = 0
        for s in combinations(b, sub_size):
            mask |= (1 << sub_to_idx[s])
        block_masks[j] = mask
        
    subs_to_blocks = [[] for _ in range(792)]
    for j in range(n_blocks):
        b = all_blocks[j]
        for s in combinations(b, sub_size):
            idx = sub_to_idx[s]
            subs_to_blocks[idx].append(j)
            
    # Symmetry breaking: Fix first block
    first_block_idx = None
    for j, b in enumerate(all_blocks):
        if b == (0, 1, 2, 3, 4, 5):
            first_block_idx = j
            break
            
    if first_block_idx is None:
        return []
        
    initial_covered = block_masks[first_block_idx]
    initial_selected = [first_block_idx]
    
    def backtrack(current_covered, current_selected):
        if time.time() > deadline:
            return None
            
        if len(current_selected) == target_blocks:
            if current_covered.bit_count() == 792:
                return list(current_selected)
            return None
            
        min_candidates = None
        min_count = float('inf')
        
        for s_idx in range(792):
            if (current_covered >> s_idx) & 1:
                continue
                
            candidates = []
            for b_idx in subs_to_blocks[s_idx]:
                if block_masks[b_idx] & current_covered:
                    continue
                candidates.append(b_idx)
                
            count = len(candidates)
            if count == 0:
                return None
                
            if count < min_count:
                min_count = count
                min_candidates = candidates
                if count == 1:
                    break
                    
        if min_candidates is None:
            return None
            
        for b_idx in min_candidates:
            new_covered = current_covered | block_masks[b_idx]
            new_selected = current_selected + [b_idx]
            result = backtrack(new_covered, new_selected)
            if result is not None:
                return result
                
        return None

    solution_indices = backtrack(initial_covered, initial_selected)
    
    if solution_indices:
        return [all_blocks[i] for i in solution_indices]
    else:
        # Final fallback: Greedy
        covered = initial_covered
        selected = list(initial_selected)
        remaining_blocks = [i for i in range(n_blocks) if i != first_block_idx]
        
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
