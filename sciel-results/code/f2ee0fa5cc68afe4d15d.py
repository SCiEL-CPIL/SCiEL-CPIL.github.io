import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    # k_to_t[i] is list of t-subset indices covered by k-subset i
    k_to_t = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            t_to_k[idx].append(i)
            k_to_t[i].append(idx)

    best_blocks = None
    best_count = float('inf')
    
    def greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_t_subsets, dtype=int)
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = np.array([len(k_to_t[i]) for i in range(num_k_subsets)], dtype=int)
        
        current_blocks = []
        used_blocks = np.zeros(num_k_subsets, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline:
                return None, False
            
            # Find max gain
            max_gain = np.max(gain)
            if max_gain == 0:
                return current_blocks, False # Should not happen
            
            # Find indices with max gain
            candidates = np.where(gain == max_gain)[0]
            candidates = candidates[used_blocks[candidates] == False] # Filter used
            
            if len(candidates) == 0:
                 # Fallback if all max gain are used (should be handled by filter)
                 candidates = np.where(gain > 0)[0]
                 candidates = candidates[used_blocks[candidates] == False]
                 if len(candidates) == 0: return current_blocks, False

            if randomize:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0]
            
            current_blocks.append(chosen_idx)
            used_blocks[chosen_idx] = True
            
            # Update uncovered and gains
            # Get t-subsets covered by chosen_idx
            ts_covered = k_to_t[chosen_idx]
            
            # Update uncovered counts
            for j in ts_covered:
                if uncovered[j] > 0:
                    uncovered[j] -= 1
                    if uncovered[j] == 0:
                        # This t-subset is now covered.
                        # Decrement gain for all blocks that cover it
                        for i in t_to_k[j]:
                            if not used_blocks[i]:
                                gain[i] -= 1
                                if gain[i] < 0: gain[i] = 0 # Safety

        return current_blocks, True

    def local_search_improve(block_indices, max_iters=5000):
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        # Use a set for O(1) removal check, but list for iteration
        # We need to track indices.
        
        # Precompute coverage counts
        cover_counts = np.zeros(num_t_subsets, dtype=int)
        for idx in current_blocks:
            for j in k_to_t[idx]:
                cover_counts[j] += 1
        
        # Map block index to position in list for fast removal? 
        # Or just use list and remove by value. Size is ~300, so O(N) removal is fine.
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Shuffle blocks to break symmetry
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]: # Iterate over a copy
                if time.time() > deadline:
                    break
                
                # Check if removable
                removable = True
                for j in k_to_t[idx]:
                    if cover_counts[j] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove
                    for j in k_to_t[idx]:
                        cover_counts[j] -= 1
                    current_blocks.remove(idx)
                    improved = True
                else:
                    # Try Swap: Find a block that covers many critical subsets
                    critical_ts = []
                    for j in k_to_t[idx]:
                        if cover_counts[j] == 1:
                            critical_ts.append(j)
                    
                    if not critical_ts:
                        continue
                        
                    # Find best replacement
                    best_gain = 0
                    best_candidate = -1
                    # Sample some candidates or check all?
                    # Checking all 75k is too slow per step.
                    # Heuristic: Check blocks that cover at least one critical ts.
                    candidate_indices = set()
                    for j in critical_ts:
                        candidate_indices.update(t_to_k[j])
                    
                    # Remove current blocks from candidates
                    current_set = set(current_blocks)
                    candidate_indices -= current_set
                    
                    if not candidate_indices:
                        continue
                        
                    # Sample up to 100 candidates to keep it fast
                    candidates_list = list(candidate_indices)
                    if len(candidates_list) > 100:
                        candidates_list = random.sample(candidates_list, 100)
                        
                    for cand_idx in candidates_list:
                        # Count how many critical_ts it covers
                        cnt = 0
                        for j in k_to_t[cand_idx]:
                            if cover_counts[j] == 1: # It covers a critical one
                                cnt += 1
                        if cnt > best_gain:
                            best_gain = cnt
                            best_candidate = cand_idx
                            
                    if best_candidate != -1 and best_gain > 0:
                        # Add best_candidate
                        for j in k_to_t[best_candidate]:
                            cover_counts[j] += 1
                        current_blocks.append(best_candidate)
                        
                        # Now try to remove idx again
                        removable = True
                        for j in k_to_t[idx]:
                            if cover_counts[j] == 1:
                                removable = False
                                break
                        if removable:
                            for j in k_to_t[idx]:
                                cover_counts[j] -= 1
                            current_blocks.remove(idx)
                            improved = True
                            
                        # Also try to remove other blocks that might be redundant now
                        # This is a simplified pass. A full re-prune might be better.
                        for other_idx in current_blocks[:]:
                            if other_idx == best_candidate: continue
                            rem = True
                            for j in k_to_t[other_idx]:
                                if cover_counts[j] == 1:
                                    rem = False
                                    break
                            if rem:
                                for j in k_to_t[other_idx]:
                                    cover_counts[j] -= 1
                                current_blocks.remove(other_idx)
                                improved = True

        return current_blocks

    # --- Main Loop ---
    num_restarts = 1000 # High number, limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Greedy
        blocks, valid = greedy_construct(randomize=True)
        if not valid:
            continue
            
        # 2. Local Search (includes pruning)
        # We skip a separate prune phase as local_search_improve does removals.
        # However, initial greedy solution might have many redundancies.
        # A quick prune before local search might help.
        
        # Quick Prune
        cover_counts = np.zeros(num_t_subsets, dtype=int)
        for idx in blocks:
            for j in k_to_t[idx]:
                cover_counts[j] += 1
        
        pruned_blocks = []
        for idx in blocks:
            removable = True
            for j in k_to_t[idx]:
                if cover_counts[j] == 1:
                    removable = False
                    break
            if removable:
                for j in k_to_t[idx]:
                    cover_counts[j] -= 1
            else:
                pruned_blocks.append(idx)
        
        if len(pruned_blocks) < best_count:
            ls_blocks = local_search_improve(pruned_blocks)
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = [list(k_subsets[i]) for i in ls_blocks]
        else:
            if len(pruned_blocks) < best_count:
                best_count = len(pruned_blocks)
                best_blocks = [list(k_subsets[i]) for i in pruned_blocks]

    if best_blocks is None:
        # Fallback
        blocks, valid = greedy_construct(randomize=False)
        if valid:
            best_blocks = [list(k_subsets[i]) for i in blocks]
        else:
            best_blocks = []

    return best_blocks
