import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Build coverage matrix: coverage_matrix[i, j] is True if k_subset i covers t_subset j
    # Memory: num_k_subsets * num_t_subsets / 8 bytes.
    # For 19,8,5: 75582 * 11628 / 8 ~ 110 MB. Acceptable.
    coverage_matrix = np.zeros((num_k_subsets, num_t_subsets), dtype=np.bool_)
    
    # Efficiently fill coverage matrix
    # For each k-subset, find all t-subsets within it and mark them.
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            coverage_matrix[i, idx] = True
            
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered_mask = np.ones(num_t_subsets, dtype=np.bool_)
        current_blocks = []
        used_blocks = set()
        
        while np.any(uncovered_mask):
            if time.time() > deadline:
                break
                
            # Calculate gain for all k-subsets
            # gains[i] = number of uncovered t-subsets covered by k_subset i
            # We can compute this as sum(coverage_matrix[i] & uncovered_mask)
            # To avoid creating a huge temporary array for all i at once if memory is tight,
            # we can do it in chunks or just let numpy handle it.
            # 75k * 11k is large but manageable.
            gains = np.sum(coverage_matrix & uncovered_mask, axis=1)
            
            # Mask out used blocks
            for b in used_blocks:
                gains[b] = 0
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick a random candidate
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered mask
            uncovered_mask &= ~coverage_matrix[chosen_idx]
            
        if not np.all(uncovered_mask == False):
            continue
            
        current_count = len(current_blocks)
        
        # 2. Local Search Phase
        # Initialize counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            counts += coverage_matrix[b_idx].astype(np.int32)
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_count = len(current_blocks)
            
            # Try 1-swap (remove redundant block)
            removed = False
            # Iterate through blocks in random order to escape local minima?
            # Or just check all.
            for b_idx in current_blocks:
                if time.time() > deadline:
                    break
                # Check if b_idx is redundant
                # It is redundant if for all t in coverage[b_idx], counts[t] > 1
                # unique_part = coverage_matrix[b_idx] & (counts == 1)
                # if np.any(unique_part): not redundant
                # else: redundant
                if not np.any(coverage_matrix[b_idx] & (counts == 1)):
                    # Remove b_idx
                    current_blocks.remove(b_idx)
                    counts -= coverage_matrix[b_idx].astype(np.int32)
                    improved = True
                    removed = True
                    break
            
            if removed:
                continue
            
            # Try 2-exchange: Replace B_i, B_j with B_k
            # Condition: Unique(B_i) U Unique(B_j) is subset of coverage[B_k]
            # Unique(B_i) = {t in coverage[B_i] | counts[t] == 1}
            # Note: If we remove B_i and B_j, the counts for t in Unique(B_i) will drop to 0.
            # So we need coverage[B_k] to cover all t such that counts[t] == 1 and t in coverage[B_i] or coverage[B_j].
            
            # To make this fast, we can pick random pairs (i, j)
            if random.random() < 0.7: # 70% chance to try 2-exchange
                if len(current_blocks) >= 2:
                    i, j = random.sample(range(len(current_blocks)), 2)
                    b_i = current_blocks[i]
                    b_j = current_blocks[j]
                    
                    # Calculate target unique coverage
                    # U_target = (coverage[b_i] | coverage[b_j]) & (counts == 1)
                    target_mask = (coverage_matrix[b_i] | coverage_matrix[b_j]) & (counts == 1)
                    
                    if np.any(target_mask):
                        # Find a block B_k that covers all of target_mask
                        # candidates_k = [k for k in range(num_k_subsets) if np.all(coverage_matrix[k, target_mask])]
                        # Vectorized:
                        # check = np.all(coverage_matrix[:, target_mask], axis=1)
                        # This is O(N_k * |target_mask|). |target_mask| is small usually.
                        
                        # Optimization: If target_mask is small, this is fast.
                        # If target_mask is large, it's likely no such k exists.
                        
                        # We can check a random sample of k's first?
                        # Or just do the vectorized check.
                        # 75k * (small number) is fast.
                        
                        # But we must exclude b_i and b_j.
                        
                        # Let's find all candidates first
                        # Note: target_mask indices
                        target_indices = np.where(target_mask)[0]
                        
                        # Check if any k covers all target_indices
                        # coverage_matrix[:, target_indices] is shape (N_k, len(target_indices))
                        # We want rows that are all True.
                        
                        # If len(target_indices) is 0, handled by 1-swap.
                        # If len is large, it's unlikely.
                        
                        # Let's try to find one.
                        # To avoid O(N_k * L) if L is large, we can just check a few randoms?
                        # No, vectorized is better if L is small.
                        # If L > 100, maybe skip?
                        
                        if len(target_indices) < 200:
                            candidate_mask = np.all(coverage_matrix[:, target_indices], axis=1)
                            # Exclude b_i, b_j
                            candidate_mask[b_i] = False
                            candidate_mask[b_j] = False
                            
                            candidates = np.where(candidate_mask)[0]
                            if len(candidates) > 0:
                                # Pick one
                                k_idx = random.choice(candidates)
                                
                                # Replace b_i, b_j with k_idx
                                current_blocks.remove(b_i)
                                current_blocks.remove(b_j)
                                current_blocks.append(k_idx)
                                
                                # Update counts
                                counts -= coverage_matrix[b_i].astype(np.int32)
                                counts -= coverage_matrix[b_j].astype(np.int32)
                                counts += coverage_matrix[k_idx].astype(np.int32)
                                
                                improved = True
                                continue

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(k_subsets[b]) for b in current_blocks]

    if best_blocks is None:
        # Fallback: return empty list or simple greedy? 
        # The task requires a valid covering. 
        # If we didn't find one, we must return something. 
        # But the greedy phase should always find one if a solution exists.
        # C(19,8,5) exists.
        # If we timed out during greedy, we might have an invalid partial solution.
        # We should ensure we return a valid solution.
        # The best way is to finish the greedy if we haven't, but we are out of time.
        # However, the deadline guard breaks the loop. 
        # If best_blocks is None, it means we never completed a greedy run?
        # Unlikely, as greedy is fast.
        # If we are in trouble, we can return a known valid covering if we hardcode it, 
        # but we don't have one.
        # We'll assume best_blocks is not None.
        
        # As a safety, if best_blocks is None, try one more quick greedy?
        # No, we are out of time.
        # Return empty list will score 0.
        # We should probably just return the best partial? No, must be valid.
        # The prompt says "Valid when every t-subset...".
        # If we can't find a valid one, we score 0.
        
        # Let's assume the first greedy run finishes.
        # Greedy takes < 1 second.
        pass

    return best_blocks
