import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and build coverage matrix
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
    
    # Build coverage matrix: cov_matrix[b, s] = 1 if block b covers t-subset s
    cov_matrix = np.zeros((n_blocks, n_t), dtype=np.uint8)
    for b, covers in enumerate(block_covers):
        for s in covers:
            cov_matrix[b, s] = 1
    
    # Build sub_to_blocks: for each t-subset, list of blocks that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    for b in range(n_blocks):
        for s in block_covers[b]:
            sub_to_blocks[s].append(b)
            
    if not _deadline_ok(deadline):
        return []
    
    # 3. Greedy Set Cover with vectorized gain computation
    covered = np.zeros(n_t, dtype=np.uint8)  # 1 if covered
    selected = []
    selected_set = set()
    
    while not np.all(covered):
        if not _deadline_ok(deadline):
            break
        
        # Find uncovered subsets
        uncovered_mask = 1 - covered
        uncovered_count = int(np.sum(uncovered_mask))
        
        if uncovered_count == 0:
            break
        
        # Find the most constrained uncovered subset (fewest candidate blocks)
        # For each uncovered subset, count how many unselected blocks cover it
        # We need to count blocks that cover s AND are not yet selected
        # Since selected blocks already have their subsets covered, we can just count
        # from sub_to_blocks but exclude selected ones
        
        # Efficient approach: for each uncovered s, the number of available blocks
        # is len(sub_to_blocks[s]) minus number of selected blocks covering s
        # But since all selected blocks' subsets are covered, if s is uncovered,
        # no selected block covers s. So available = len(sub_to_blocks[s])
        
        # Find uncovered subset with minimum available blocks
        min_avail = float('inf')
        target_sub = -1
        for s in range(n_t):
            if uncovered_mask[s]:
                avail = len(sub_to_blocks[s])
                if avail < min_avail:
                    min_avail = avail
                    target_sub = s
                    if avail <= 1:
                        break
        
        if target_sub == -1:
            break
        
        # Among blocks covering target_sub, pick the one with max gain
        candidates = sub_to_blocks[target_sub]
        
        # Compute gain for each candidate: number of currently uncovered subsets it covers
        # gain[b] = sum(cov_matrix[b, s] * uncovered_mask[s] for s in range(n_t))
        # Vectorized: gain = cov_matrix[candidates, :] @ uncovered_mask
        candidate_indices = np.array(candidates)
        gains = cov_matrix[candidate_indices, :] @ uncovered_mask
        
        best_idx = np.argmax(gains)
        best_block = int(candidate_indices[best_idx])
        max_gain = int(gains[best_idx])
        
        if max_gain <= 0:
            break
        
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update covered
        covered[cov_matrix[best_block, :]] = 1
    
    if not _deadline_ok(deadline):
        result = [list(blocks[i]) for i in selected]
        return result
    
    # 4. Prune redundant blocks
    final_counts = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        final_counts[cov_matrix[i, :]] += 1
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            # Check if block i is redundant: all its subsets have count > 1
            block_subsets = block_covers[i]
            if all(final_counts[s] > 1 for s in block_subsets):
                selected.remove(i)
                selected_set.discard(i)
                final_counts[cov_matrix[i, :]] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    
    if not _deadline_ok(deadline):
        result = [list(blocks[i]) for i in selected]
        return result
    
    # 5. Local search: try to reduce block count via swaps
    # For each block in selected, try removing it and see if we can add fewer blocks to restore coverage
    # Simpler: try removing each block one at a time, and if coverage is lost,
    # try to find a single block that covers all the newly uncovered subsets
    
    # Build coverage from current selection
    cov_from_selected = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        cov_from_selected[cov_matrix[i, :]] += 1
    
    improved = True
    max_iterations = 200
    iteration = 0
    while improved and _deadline_ok(deadline) and iteration < max_iterations:
        improved = False
        iteration += 1
        
        for idx, i in enumerate(list(selected)):
            if not _deadline_ok(deadline):
                break
            
            # Try removing block i
            # Which subsets become uncovered?
            newly_uncovered = []
            for s in block_covers[i]:
                if cov_from_selected[s] <= 1:
                    newly_uncovered.append(s)
            
            if not newly_uncovered:
                # Block i is redundant
                selected.remove(i)
                selected_set.discard(i)
                cov_from_selected[cov_matrix[i, :]] -= 1
                improved = True
                continue
            
            # Find a block that covers all newly_uncovered subsets
            # and has the highest total coverage of currently uncovered + newly_uncovered
            # Actually, we just need to find any block that covers all newly_uncovered
            # and minimizes additional blocks needed (ideally 1)
            
            # Get intersection of sub_to_blocks for all newly_uncovered
            if len(newly_uncovered) == 1:
                candidates = sub_to_blocks[newly_uncovered[0]]
            else:
                # Find blocks that cover ALL newly_uncovered
                candidate_set = set(sub_to_blocks[newly_uncovered[0]])
                for s in newly_uncovered[1:]:
                    candidate_set = candidate_set.intersection(sub_to_blocks[s])
                candidates = list(candidate_set)
            
            if not candidates:
                continue
            
            # Among candidates, pick the one that maximizes coverage of all currently uncovered
            # subsets (to minimize future need)
            current_uncovered = np.where(cov_from_selected == 0)[0]
            if len(current_uncovered) == 0:
                # No uncovered subsets, just pick any candidate
                best_cand = candidates[0]
            else:
                cand_arr = np.array(candidates)
                # Gain = how many of current_uncovered this block covers
                gains = cov_matrix[cand_arr, current_uncovered] @ np.ones(len(current_uncovered))
                best_idx = np.argmax(gains)
                best_cand = int(cand_arr[best_idx])
            
            # Check: does best_cand cover all newly_uncovered?
            covers_all = all(cov_matrix[best_cand, s] == 1 for s in newly_uncovered)
            if not covers_all:
                continue
            
            # Swap: remove i, add best_cand
            # But we need to make sure best_cand is not already in selected
            if best_cand in selected_set:
                continue
            
            # Perform swap
            selected.remove(i)
            selected_set.discard(i)
            cov_from_selected[cov_matrix[i, :]] -= 1
            
            selected.append(best_cand)
            selected_set.add(best_cand)
            cov_from_selected[cov_matrix[best_cand, :]] += 1
            
            improved = True
    
    # Final prune
    final_counts2 = np.zeros(n_t, dtype=np.int32)
    for i in selected:
        final_counts2[cov_matrix[i, :]] += 1
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            block_subsets = block_covers[i]
            if all(final_counts2[s] > 1 for s in block_subsets):
                selected.remove(i)
                selected_set.discard(i)
                final_counts2[cov_matrix[i, :]] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    
    result = [list(blocks[i]) for i in selected]
    
    return result
