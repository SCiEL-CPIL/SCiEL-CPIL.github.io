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
    
    def greedy_construct(rng_seed=0):
        """Run a randomized greedy construction with tie-breaking."""
        rng = np.random.default_rng(rng_seed)
        covered = np.zeros(num_tsubs, dtype=bool)
        selected_set = set()
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        
        while time.monotonic() < deadline - 15.0:
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
                # Randomized tie-breaking: prefer blocks with less overlap with selected
                # but add randomness to explore different solutions
                overlaps = np.array([
                    np.sum(cover_count[block_tsub_idxs[c]] > 0) 
                    for c in candidates
                ])
                # Add small random perturbation to break ties
                perturbation = rng.random(len(overlaps)) * 0.1
                scores = overlaps + perturbation
                best_idx = candidates[np.argmin(scores)]
            
            selected_set.add(best_idx)
            
            # Update covered and cover_count
            for idx in block_tsub_idxs[best_idx]:
                cover_count[idx] += 1
                covered[idx] = True
                
            if np.all(covered):
                break
                
        return selected_set, cover_count
    
    def polish(selected_set, cover_count):
        """Remove redundant blocks."""
        counts = cover_count.copy()
        changed = True
        while changed and time.monotonic() < deadline - 10.0:
            changed = False
            for b_idx in sorted(selected_set, reverse=True):
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
        return selected_set, counts
    
    def local_search(selected_set, counts):
        """Try to improve by swapping blocks."""
        if time.monotonic() > deadline - 20.0:
            return selected_set, counts
            
        selected_arr = list(selected_set)
        non_selected = [i for i in range(num_kblocks) if i not in selected_set]
        
        improved = True
        while improved and time.monotonic() < deadline - 15.0:
            improved = False
            for pos in range(len(selected_arr)):
                if time.monotonic() > deadline - 15.0:
                    break
                b_remove = selected_arr[pos]
                critical_idxs = [idx for idx in block_tsub_idxs[b_remove] if counts[idx] == 1]
                
                if not critical_idxs:
                    selected_arr.pop(pos)
                    for idx in block_tsub_idxs[b_remove]:
                        counts[idx] -= 1
                    improved = True
                    break
                
                best_new = -1
                best_new_gain = -1
                for new_b in non_selected:
                    new_tsubs = set(block_tsub_idxs[new_b])
                    if not all(ci in new_tsubs for ci in critical_idxs):
                        continue
                    new_gain = sum(1 for idx in block_tsub_idxs[new_b] if counts[idx] == 0)
                    if new_gain > best_new_gain:
                        best_new_gain = new_gain
                        best_new = new_b
                
                if best_new >= 0:
                    for idx in block_tsub_idxs[b_remove]:
                        counts[idx] -= 1
                    for idx in block_tsub_idxs[best_new]:
                        counts[idx] += 1
                    selected_arr[pos] = best_new
                    non_selected.remove(best_new)
                    non_selected.append(b_remove)
                    improved = True
                    break
        
        return set(selected_arr), counts
    
    # 3. Multi-restart: try several different random seeds
    best_result = None
    best_count = float('inf')
    
    # First run: deterministic (seed=0)
    sel, cnt = greedy_construct(0)
    sel, cnt = polish(sel, cnt)
    sel, cnt = local_search(sel, cnt)
    best_result = sel
    best_count = len(sel)
    
    # Additional restarts with different seeds
    num_restarts = 5
    for seed in range(1, num_restarts + 1):
        if time.monotonic() > deadline - 25.0:
            break
        sel, cnt = greedy_construct(seed)
        sel, cnt = polish(sel, cnt)
        sel, cnt = local_search(sel, cnt)
        if len(sel) < best_count:
            best_result = sel
            best_count = len(sel)
    
    # 4. Final polish
    if time.monotonic() < deadline - 5.0:
        counts_final = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in best_result:
            for idx in block_tsub_idxs[b_idx]:
                counts_final[idx] += 1
        best_result, counts_final = polish(best_result, counts_final)
    
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in sorted(best_result)]
    
    return result_blocks
