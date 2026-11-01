import itertools
import time
import random
import numpy as np
from collections import defaultdict

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

    def prune_redundant(block_indices):
        """
        Remove redundant blocks from a valid covering.
        A block is redundant if every t-subset it covers is also covered by another block.
        We do this in a single pass, updating counts as we go.
        """
        if not block_indices:
            return []
        
        cover_counts = np.zeros(num_t_subsets, dtype=int)
        # First, count all coverings
        for idx in block_indices:
            for j in k_to_t[idx]:
                cover_counts[j] += 1
        
        pruned = []
        # To make it more effective, we can process in a specific order, but random order is fine.
        # Actually, the standard greedy removal is:
        # For each block, check if any t-subset has count 1. If so, keep it.
        # If not, remove it and decrement counts.
        
        # We need to be careful: if we remove a block, counts change, which might make
        # previously "essential" blocks redundant.
        # So we iterate until no more removals are possible, or just one pass?
        # One pass is faster but might miss some. Multiple passes is better.
        # Given time constraints, a few passes are good.
        
        changed = True
        # We work on a copy of the list
        current_list = list(block_indices)
        
        while changed and time.time() < deadline:
            changed = False
            # Recalculate counts? Or update incrementally?
            # Incremental is better.
            # But we need to know which blocks are currently in the set.
            # Let's use a set for O(1) membership check if needed, but we have indices.
            
            # To be safe and simple, let's just do a single pass first.
            # If we have time, we can do more.
            
            # Actually, the standard "greedy set cover" removal works best if we
            # remove blocks that are "most redundant" first? Or just any?
            # Let's just scan and remove.
            
            new_list = []
            # We need to re-count for the current set?
            # If we maintain cover_counts incrementally, we can do it in one pass.
            # But cover_counts reflects the state BEFORE removals in this pass.
            # So if we remove block A, and block B was only covered by A and B,
            # B is now critical. But we already decided to keep B (if count was 1).
            # So single pass is safe (never removes a critical block).
            # It might not remove all possible redundancies in one pass.
            
            # Let's just do one pass for speed, and rely on local search for more.
            
            # Re-count for the current set to be sure?
            # No, let's use the incremental approach from the start.
            # But the input to prune_redundant is a list.
            # Let's just re-count from scratch for simplicity, as num_blocks is small (~300-400).
            cover_counts = np.zeros(num_t_subsets, dtype=int)
            for idx in current_list:
                for j in k_to_t[idx]:
                    cover_counts[j] += 1
            
            for idx in current_list:
                if time.time() > deadline:
                    # Return what we have so far
                    break
                
                removable = True
                for j in k_to_t[idx]:
                    if cover_counts[j] == 1:
                        removable = False
                        break
                
                if removable:
                    for j in k_to_t[idx]:
                        cover_counts[j] -= 1
                    changed = True
                else:
                    new_list.append(idx)
            
            current_list = new_list
            
            # If no change, we are done.
            # If we want more passes, we can loop.
            # Let's do a few passes if we have time.
            if not changed:
                break
            
            # Optional: check time for next pass
            if time.time() > deadline:
                break

        return current_list

    def local_search_improve(block_indices, max_iters=2000):
        """
        Local search: try to swap blocks to reduce count.
        1. Try to remove a block.
        2. If not removable, try to swap it with another block that covers its critical subsets.
        """
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        
        # Precompute coverage counts
        cover_counts = np.zeros(num_t_subsets, dtype=int)
        for idx in current_blocks:
            for j in k_to_t[idx]:
                cover_counts[j] += 1
        
        # Map block index to position in list?
        # Since we swap/remove, maintaining indices is tricky.
        # We'll just use the list of indices.
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
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
                    # Try Swap
                    critical_ts = []
                    for j in k_to_t[idx]:
                        if cover_counts[j] == 1:
                            critical_ts.append(j)
                    
                    if not critical_ts:
                        continue
                        
                    # Find best replacement
                    best_gain = 0
                    best_candidate = -1
                    
                    # Candidates: blocks that cover at least one critical ts
                    candidate_indices = set()
                    for j in critical_ts:
                        candidate_indices.update(t_to_k[j])
                    
                    # Remove current blocks from candidates
                    current_set = set(current_blocks)
                    candidate_indices -= current_set
                    
                    if not candidate_indices:
                        continue
                        
                    # Sample candidates to speed up
                    candidates_list = list(candidate_indices)
                    if len(candidates_list) > 200:
                        candidates_list = random.sample(candidates_list, 200)
                        
                    for cand_idx in candidates_list:
                        # Count how many critical_ts it covers
                        cnt = 0
                        for j in k_to_t[cand_idx]:
                            if cover_counts[j] == 1:
                                cnt += 1
                        if cnt > best_gain:
                            best_gain = cnt
                            best_candidate = cand_idx
                            
                    if best_candidate != -1 and best_gain > 0:
                        # Perform swap
                        # Add best_candidate
                        for j in k_to_t[best_candidate]:
                            cover_counts[j] += 1
                        
                        # Now try to remove idx
                        removable = True
                        for j in k_to_t[idx]:
                            if cover_counts[j] == 1:
                                removable = False
                                break
                        
                        if removable:
                            for j in k_to_t[idx]:
                                cover_counts[j] -= 1
                            current_blocks.remove(idx)
                            current_blocks.append(best_candidate)
                            improved = True
                            
                            # After a swap, some other blocks might become redundant.
                            # We don't check them immediately, but the next iteration will.
                            # Or we can do a quick prune pass here?
                            # Let's let the main loop handle it.

        return current_blocks

    # --- Main Loop ---
    num_restarts = 500 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # 1. Greedy
        blocks, valid = greedy_construct(randomize=True)
        if not valid:
            continue
            
        # 2. Prune
        pruned_blocks = prune_redundant(blocks)
        
        # 3. Local Search
        ls_blocks = local_search_improve(pruned_blocks)
        
        # 4. Final Prune (local search might leave redundancies)
        final_blocks = prune_redundant(ls_blocks)
        
        if len(final_blocks) < best_count:
            best_count = len(final_blocks)
            best_blocks = [list(k_subsets[i]) for i in final_blocks]
            # Optional: print progress
            # print(f"Restart {restart}: New best {best_count}")

    if best_blocks is None:
        # Fallback
        blocks, valid = greedy_construct(randomize=False)
        if valid:
            best_blocks = [list(k_subsets[i]) for i in blocks]
        else:
            best_blocks = []

    return best_blocks
