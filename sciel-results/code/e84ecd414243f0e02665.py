import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all t-subsets
    all_tsubs = list(itertools.combinations(points, t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Generate all k-blocks
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # Build dense incidence matrix A (num_tsubs x num_kblocks)
    # A[i, j] = 1 if tsub i is in kblock j
    # Size: ~3876 x 50388 for v=19, k=7, t=4. 
    # 3876 * 50388 * 1 byte (bool) ~ 195 MB. This is acceptable.
    # We use uint8 for fast summing.
    A = np.zeros((num_tsubs, num_kblocks), dtype=np.uint8)
    
    # Precompute t-subset indices for each block for fast updates
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = np.array([tsub_to_idx[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_tsub_idxs.append(idxs)
        # Set A[i, j] = 1 for all i in idxs
        A[idxs, j] = 1

    best_blocks = None
    best_size = float('inf')
    
    def greedy_construction(random_seed=0):
        rng = np.random.RandomState(random_seed)
        
        covered = np.zeros(num_tsubs, dtype=bool)
        selected_blocks = []
        selected_set = set()
        
        # Greedy phase
        while time.monotonic() < deadline - 5.0:
            u = ~covered
            if not np.any(u):
                break
                
            # Calculate gains: for each block j, gain[j] = sum(A[i, j] for i where u[i]=True)
            # gains = A.T @ u  (sparse would be better if large, but dense is fast for this size)
            gains = np.dot(A.T, u.astype(np.uint8))
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            if len(candidates) == 1:
                best_idx = candidates[0]
            else:
                # Tie-breaking: Random choice or heuristic
                # Heuristic: Prefer block with less overlap with already selected (hard to compute fast)
                # Simple: Random choice among candidates
                best_idx = candidates[rng.randint(len(candidates))]
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update covered
            covered[block_tsub_idxs[best_idx]] = True
            
            if np.all(covered):
                break
                
        # Polish: Remove redundant blocks
        selected_blocks = polish(selected_blocks, block_tsub_idxs, A)
        
        # Local Search: 1-Swap
        selected_blocks = local_search_1swap(selected_blocks, block_tsub_idxs, A, deadline)
        
        # Final Polish
        selected_blocks = polish(selected_blocks, block_tsub_idxs, A)
        
        return selected_blocks

    def polish(selected_blocks, block_tsub_idxs, A):
        selected_set = set(selected_blocks)
        # Count coverage
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected_set:
            counts[block_tsub_idxs[b_idx]] += 1
            
        changed = True
        while changed and time.monotonic() < deadline - 2.0:
            changed = False
            # Try removing blocks in reverse order of selection or random order
            for b_idx in list(selected_set):
                if b_idx not in selected_set:
                    continue
                can_remove = True
                for idx in block_tsub_idxs[b_idx]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    selected_set.discard(b_idx)
                    counts[block_tsub_idxs[b_idx]] -= 1
                    changed = True
        return sorted(selected_set)

    def local_search_1swap(selected_blocks, block_tsub_idxs, A, deadline):
        selected_set = set(selected_blocks)
        non_selected = [i for i in range(num_kblocks) if i not in selected_set]
        
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for b_idx in selected_set:
            counts[block_tsub_idxs[b_idx]] += 1
            
        improved = True
        while improved and time.monotonic() < deadline - 2.0:
            improved = False
            for pos, b_remove in enumerate(list(selected_set)):
                if time.monotonic() > deadline - 2.0:
                    break
                
                # Find critical t-subsets (covered only by b_remove)
                critical_idxs = []
                for idx in block_tsub_idxs[b_remove]:
                    if counts[idx] == 1:
                        critical_idxs.append(idx)
                
                if not critical_idxs:
                    # Can remove without replacement
                    selected_set.discard(b_remove)
                    counts[block_tsub_idxs[b_remove]] -= 1
                    non_selected.append(b_remove)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all critical t-subsets
                # and minimizes the number of "wasted" covers (or maximizes new coverage)
                best_new = -1
                best_score = -1
                
                # Optimization: Only consider blocks that cover at least one critical idx
                # We can pre-filter or just check all non_selected
                # To speed up, we can check if the intersection of critical_idxs and block_tsub_idxs[new_b] is critical_idxs
                for new_b in non_selected:
                    new_tsubs = block_tsub_idxs[new_b]
                    # Check if new_b covers all critical_idxs
                    # This is a subset check. 
                    # Efficient way: sum(A[critical_idxs, new_b]) == len(critical_idxs)
                    if np.sum(A[critical_idxs, new_b]) == len(critical_idxs):
                        # Score: number of NEW t-subsets covered by new_b (where count was 0)
                        # But we are replacing b_remove, so we need to be careful with counts.
                        # If we swap b_remove -> new_b:
                        # Counts for tsubs in new_b \ b_remove increase by 1 (if they were 0, they become 1)
                        # Counts for tsubs in b_remove \ new_b decrease by 1
                        # We want to minimize the number of tsubs that become uncovered? No, we already checked criticals.
                        # We want to maximize the "efficiency". 
                        # A good metric: number of tsubs in new_b that are currently covered 0 times.
                        new_gain = np.sum(counts[new_tsubs] == 0)
                        if new_gain > best_score:
                            best_score = new_gain
                            best_new = new_b
                
                if best_new >= 0:
                    # Perform swap
                    selected_set.discard(b_remove)
                    selected_set.add(best_new)
                    counts[block_tsub_idxs[b_remove]] -= 1
                    counts[block_tsub_idxs[best_new]] += 1
                    non_selected.remove(best_new)
                    non_selected.append(b_remove)
                    improved = True
                    break
                    
        return sorted(selected_set)

    # Run multiple restarts
    num_restarts = 5
    for i in range(num_restarts):
        if time.monotonic() > deadline - 10.0:
            break
        blocks = greedy_construction(random_seed=i)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: Just greedy
        best_blocks = greedy_construction(random_seed=0)
        
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in best_blocks]
    return result_blocks
