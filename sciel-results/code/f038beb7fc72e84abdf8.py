import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline):
    """Remove redundant blocks from selected set."""
    if not selected:
        return set()
    
    # Calculate coverage counts
    final_counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    # Try to remove blocks iteratively
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant
            redundant = True
            for s in block_covers[i]:
                if final_counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    final_counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    return selected

def _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline):
    """Local search: try to replace blocks with better alternatives."""
    if not selected:
        return selected
    
    # Calculate coverage counts
    final_counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        for s in block_covers[i]:
            final_counts[s] += 1
    
    best_selected = set(selected)
    best_size = len(selected)
    
    # Precompute block_covers as sets for fast membership testing
    block_covers_set = [set(c) for c in block_covers]
    
    # Pass 1: Try removing one block and adding another (1-swap)
    for b_remove in list(selected):
        if not _deadline_ok(deadline):
            break
        
        # Find subsets only covered by b_remove
        unique_subs = []
        for s in block_covers[b_remove]:
            if final_counts[s] == 1:
                unique_subs.append(s)
        
        if not unique_subs:
            # Block is redundant, remove it
            new_selected = set(selected)
            new_selected.discard(b_remove)
            if len(new_selected) < best_size:
                best_selected = new_selected
                best_size = len(new_selected)
            continue
        
        # Try to find a single block that covers all unique_subs
        if len(unique_subs) > 3:
            continue
        
        # Find candidate blocks that cover all unique_subs
        candidate_blocks = None
        for s in unique_subs:
            if candidate_blocks is None:
                candidate_blocks = set(sub_to_blocks[s])
            else:
                candidate_blocks &= set(sub_to_blocks[s])
        
        if not candidate_blocks:
            continue
        
        # Try each candidate
        for b_add in candidate_blocks:
            if b_add in selected:
                continue
            
            # Check if adding b_add and removing b_remove keeps all subsets covered
            valid = True
            covers_add = block_covers_set[b_add]
            for s in block_covers[b_remove]:
                if final_counts[s] == 1 and s not in covers_add:
                    valid = False
                    break
            
            if valid:
                new_selected = set(selected)
                new_selected.discard(b_remove)
                new_selected.add(b_add)
                if len(new_selected) < best_size:
                    best_selected = new_selected
                    best_size = len(new_selected)
    
    # Pass 2: Try removing two blocks and adding one (2-swap)
    if _deadline_ok(deadline):
        # Rebuild counts from best_selected
        final_counts2 = np.zeros(n_t, dtype=np.int32)
        for i in best_selected:
            for s in block_covers[i]:
                final_counts2[s] += 1
        
        selected_list = list(best_selected)
        n_sel = len(selected_list)
        
        # Limit pairs to keep runtime manageable
        max_pairs = min(n_sel * (n_sel - 1) // 2, 500)
        pair_count = 0
        
        for i_idx in range(n_sel):
            if not _deadline_ok(deadline) or pair_count >= max_pairs:
                break
            for j_idx in range(i_idx + 1, n_sel):
                if not _deadline_ok(deadline) or pair_count >= max_pairs:
                    break
                pair_count += 1
                
                b1 = selected_list[i_idx]
                b2 = selected_list[j_idx]
                
                # Find subsets only covered by b1 or b2 (i.e., count == 1 after removing both)
                # A subset s is "uniquely covered by {b1,b2}" if final_counts2[s] <= 2 and 
                # (s in block_covers[b1] or s in block_covers[b2]) and 
                # the count from b1+b2 is exactly final_counts2[s]
                
                # Simpler: find all subsets that would become uncovered if we remove b1 and b2
                # i.e., subsets where final_counts2[s] == 1 and s is in b1 or b2,
                # OR final_counts2[s] == 2 and s is in both b1 and b2
                unique_subs = []
                covers1 = block_covers_set[b1]
                covers2 = block_covers_set[b2]
                
                for s in covers1:
                    if final_counts2[s] == 1:
                        unique_subs.append(s)
                
                for s in covers2:
                    if final_counts2[s] == 1:
                        unique_subs.append(s)
                    elif final_counts2[s] == 2 and s in covers1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    # Both blocks are redundant
                    new_selected = set(best_selected)
                    new_selected.discard(b1)
                    new_selected.discard(b2)
                    if len(new_selected) < best_size:
                        best_selected = new_selected
                        best_size = len(new_selected)
                    continue
                
                # Try to find a single block that covers all unique_subs
                if len(unique_subs) > 4:
                    continue
                
                candidate_blocks = None
                for s in unique_subs:
                    if candidate_blocks is None:
                        candidate_blocks = set(sub_to_blocks[s])
                    else:
                        candidate_blocks &= set(sub_to_blocks[s])
                
                if not candidate_blocks:
                    continue
                
                for b_add in candidate_blocks:
                    if b_add in best_selected:
                        continue
                    
                    # Check if adding b_add and removing b1, b2 keeps all subsets covered
                    valid = True
                    covers_add = block_covers_set[b_add]
                    
                    for s in covers1:
                        # After removing b1 and b2, count of s is final_counts2[s] - (1 if s in covers1) - (1 if s in covers2)
                        cnt_after = final_counts2[s]
                        if s in covers1:
                            cnt_after -= 1
                        if s in covers2:
                            cnt_after -= 1
                        if cnt_after == 0 and s not in covers_add:
                            valid = False
                            break
                    
                    if valid:
                        for s in covers2:
                            if s in covers1:
                                continue  # already checked
                            cnt_after = final_counts2[s]
                            if s in covers1:
                                cnt_after -= 1
                            if s in covers2:
                                cnt_after -= 1
                            if cnt_after == 0 and s not in covers_add:
                                valid = False
                                break
                    
                    if valid:
                        new_selected = set(best_selected)
                        new_selected.discard(b1)
                        new_selected.discard(b2)
                        new_selected.add(b_add)
                        if len(new_selected) < best_size:
                            best_selected = new_selected
                            best_size = len(new_selected)
    
    return best_selected

def _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng=None):
    """Greedy set cover with randomized tie-breaking, vectorized."""
    if rng is None:
        rng = random.Random()
    
    n_blocks = len(blocks)
    
    # Build incidence matrix: block_covers as a sparse-like structure
    # For speed, precompute block_covers as a list of arrays
    block_covers_arr = [np.array(c, dtype=np.int32) for c in block_covers]
    
    # sub_to_blocks as list of arrays
    sub_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in sub_to_blocks]
    
    # uncovered as a boolean array
    uncovered = np.ones(n_t, dtype=bool)
    uncovered_count = n_t
    selected = set()
    
    while uncovered_count > 0:
        if not _deadline_ok(deadline):
            break
            
        # Find the uncovered subset with fewest candidate blocks
        # Only consider uncovered subsets
        min_count = n_blocks + 1
        target_sub = -1
        for s in range(n_t):
            if uncovered[s]:
                c = len(sub_to_blocks_arr[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
        
        if target_sub == -1:
            break
            
        # Get candidates for target_sub
        candidates = sub_to_blocks_arr[target_sub]
        
        # Pick the candidate that maximizes coverage of other uncovered subsets
        # Vectorized: for each candidate block, count how many of its covered subsets are uncovered
        best_block = -1
        max_gain = -1
        
        # For each candidate, compute gain = number of uncovered subsets it covers
        # We can vectorize this if we have a matrix, but let's do it semi-vectorized
        for b in candidates:
            covers = block_covers_arr[b]
            gain = np.sum(uncovered[covers])
            if gain > max_gain:
                max_gain = gain
                best_block = b
            elif gain == max_gain and gain > 0:
                # Random tie-breaking
                if rng.random() < 0.5:
                    best_block = b
                    max_gain = gain
        
        # Select best_block
        selected.add(best_block)
        
        # Update uncovered
        covers = block_covers_arr[best_block]
        newly_covered = uncovered[covers]
        uncovered_count -= np.sum(newly_covered)
        uncovered[covers] = False
    
    return selected

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []
    
    # 3. Multiple randomized greedy restarts with local search
    best_result = None
    best_size = float('inf')
    
    # Number of restarts depends on available time
    elapsed = time.time() - start
    remaining = deadline - time.time()
    # Estimate time per restart: greedy is faster now with numpy, local search adds more
    # Let's try more restarts with the speedup
    n_restarts = max(5, int(remaining / 15))  # ~15s per restart estimate
    n_restarts = min(n_restarts, 15)  # Cap at 15
    
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline, rng)
        
        if not selected:
            continue
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        if _deadline_ok(deadline):
            selected = _local_search(selected, blocks, block_covers, sub_to_blocks, n_t, deadline)
        
        # Check if this is better
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = set(selected)
        
        # Early stopping if we hit a very good solution
        if best_size <= 270:
            break
    
    # If no result found, fall back to a single greedy
    if best_result is None:
        selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        best_result = selected
    
    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity (just in case)
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
            # Invalid, try again with a fresh greedy
            selected = _greedy_cover(blocks, block_covers, sub_to_blocks, n_t, deadline)
            selected = _prune_redundant(selected, block_covers, n_t, deadline)
            result = [list(blocks[i]) for i in selected]
            
    return result
