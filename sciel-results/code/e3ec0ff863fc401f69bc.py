import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    # A[i, j] = 1 if tsub i is contained in kblock j
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(all_kblocks):
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # Precompute t-subset indices for each block for fast updating
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
    
    # Precompute block-block overlap counts (how many t-subsets two blocks share)
    # This helps with tie-breaking in greedy: prefer blocks that overlap less with selected
    # We'll track this dynamically instead for efficiency
    
    # 3. Greedy Construction with improved tie-breaking
    covered = np.zeros(num_tsubs, dtype=bool)
    selected_blocks = []
    selected_set = set()
    
    # Track which t-subsets are covered by how many selected blocks
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    
    while time.monotonic() < deadline - 10.0:
        u = ~covered
        gains = A.T @ u
        best_gain = gains.max()
        if best_gain == 0:
            break
        
        # Find all blocks with max gain
        max_gain_mask = (gains == best_gain)
        candidates = np.where(max_gain_mask)[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Tie-breaking: prefer block with least overlap (shared t-subsets) with already selected
            # Compute for each candidate how many of its t-subsets are already covered
            # (i.e., shared with existing selected blocks)
            best_idx = candidates[0]
            min_overlap = float('inf')
            for cand in candidates:
                # Count how many t-subsets of this block are already in cover_count > 0
                # This approximates overlap with selected blocks
                overlap = np.sum(cover_count[block_tsub_idxs[cand]] > 0)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_idx = cand
        
        selected_blocks.append(best_idx)
        selected_set.add(best_idx)
        
        # Update covered and cover_count
        for idx in block_tsub_idxs[best_idx]:
            cover_count[idx] += 1
            covered[idx] = True
            
        if np.all(covered):
            break
            
    # 4. Aggressive Polish (Remove redundant blocks) - multiple passes with different orderings
    counts = cover_count.copy()
    
    changed = True
    num_passes = 0
    while changed and time.monotonic() < deadline - 8.0:
        changed = False
        num_passes += 1
        
        # Alternate between forward and reverse order each pass
        if num_passes % 2 == 1:
            order = sorted(selected_set, reverse=True)
        else:
            order = sorted(selected_set)
        
        for b_idx in order:
            if b_idx not in selected_set:
                continue
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.discard(b_idx)
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
    
    selected_blocks = sorted(selected_set)
    
    # 5. Local search: try swapping one block for another to reduce total
    remaining = deadline - time.monotonic()
    if remaining > 10.0 and len(selected_blocks) > 0:
        selected_arr = np.array(selected_blocks, dtype=np.int64)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected_arr:
            for idx in block_tsub_idxs[b_idx]:
                counts[idx] += 1
        
        # Build a list of non-selected blocks
        non_selected = [i for i in range(num_kblocks) if i not in selected_set]
        
        improved = True
        while improved and time.monotonic() < deadline - 5.0:
            improved = False
            for pos in range(len(selected_arr)):
                if time.monotonic() > deadline - 5.0:
                    break
                b_remove = selected_arr[pos]
                # Check if removing this block leaves any uncovered t-subset
                critical_idxs = []
                for idx in block_tsub_idxs[b_remove]:
                    if counts[idx] == 1:
                        critical_idxs.append(idx)
                
                if not critical_idxs:
                    # Can remove without replacement
                    selected_arr = np.delete(selected_arr, pos)
                    for idx in block_tsub_idxs[b_remove]:
                        counts[idx] -= 1
                    improved = True
                    break
                
                # Try to find a replacement block that covers all critical t-subsets
                # and adds minimal new coverage
                best_new = -1
                best_new_gain = -1
                for new_b in non_selected:
                    # Check if new_b covers all critical t-subsets
                    new_tsubs = set(block_tsub_idxs[new_b])
                    if not all(ci in new_tsubs for ci in critical_idxs):
                        continue
                    # Count how many new t-subsets this block covers
                    new_gain = sum(1 for idx in block_tsub_idxs[new_b] if counts[idx] == 0)
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_new = new_b
                
                if best_new >= 0:
                    # Perform swap
                    for idx in block_tsub_idxs[b_remove]:
                        counts[idx] -= 1
                    for idx in block_tsub_idxs[best_new]:
                        counts[idx] += 1
                    selected_arr[pos] = best_new
                    non_selected.remove(best_new)
                    non_selected.append(b_remove)
                    improved = True
                    break
    
        selected_blocks = sorted(selected_arr.tolist())
    
    # 6. Final polish after local search
    if time.monotonic() < deadline - 3.0:
        selected_set2 = set(selected_blocks)
        counts2 = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected_set2:
            for idx in block_tsub_idxs[b_idx]:
                counts2[idx] += 1
        
        changed2 = True
        while changed2 and time.monotonic() < deadline - 2.0:
            changed2 = False
            for b_idx in sorted(selected_set2, reverse=True):
                if b_idx not in selected_set2:
                    continue
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts2[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected_set2.discard(b_idx)
                    for idx in block_tsub_idxs[b_idx]:
                        counts2[idx] -= 1
                    changed2 = True
        
        selected_blocks = sorted(selected_set2)
    
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
