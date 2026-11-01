import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using Greedy Initialization followed by Local Search (Reduction and 2-Opt Swaps).
    """
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the indices of the t-subsets it covers
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    # Precompute reverse index: for each t-subset, which k-subsets cover it?
    # This is crucial for the 2-opt "find replacement" step.
    # triple_blocks[i] = list of block indices that cover t-subset i
    triple_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, indices in enumerate(block_cover_indices):
        for t_idx in indices:
            triple_blocks[t_idx].append(b_idx)
            
    # Convert block_cover_indices to a list of sets for faster intersection/union?
    # Or just lists.
    # For "is subset" checks in 2-opt, we might need to check if a candidate block covers U_def.
    # U_def is small. We can just check if all t in U_def are in block_cover_indices[candidate].
    
    def greedy_init():
        """
        Fast randomized greedy to get an initial valid covering.
        """
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found = []
        
        while np.any(uncovered):
            # Pick a random uncovered t-subset to anchor the search
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_t_idx = uncovered_indices[random.randint(0, len(uncovered_indices)-1)]
            
            # Find blocks that cover this target t-subset
            candidates = triple_blocks[target_t_idx]
            if not candidates:
                break # Should not happen
            
            # Evaluate candidates: which covers the most *other* uncovered t-subsets?
            best_block = -1
            best_gain = -1
            
            # Sample a subset of candidates if list is too long?
            # triple_blocks[target] can be up to C(24-3, 7-3) = C(21,4) = 5985.
            # 5985 is small enough to scan fully.
            for b_idx in candidates:
                # Count how many uncovered t-subsets this block covers
                gain = 0
                for t_idx in block_cover_indices[b_idx]:
                    if uncovered[t_idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks_found.append(best_block)
            # Update uncovered
            for t_idx in block_cover_indices[best_block]:
                uncovered[t_idx] = False
                
        return blocks_found

    def is_valid_covering(block_indices):
        """
        Check if the set of block indices forms a valid covering.
        Returns True if all t-subsets are covered.
        """
        if not block_indices:
            return n_t_subsets == 0
        covered = np.zeros(n_t_subsets, dtype=bool)
        for b_idx in block_indices:
            covered[block_cover_indices[b_idx]] = True
        return np.all(covered)

    def local_search(initial_blocks, timeout):
        """
        Improve the covering using reduction and 2-opt swaps.
        """
        current_blocks = list(initial_blocks)
        # Maintain count of how many blocks cover each t-subset
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            counts[block_cover_indices[b_idx]] += 1
            
        best_blocks = current_blocks[:]
        best_size = len(current_blocks)
        
        start_time = time.time()
        
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            
            # Phase 1: Try to remove single blocks
            # Iterate through a copy of the list
            to_remove = []
            for b_idx in current_blocks:
                # Check if removing b_idx leaves all its t-subsets covered by others
                # i.e., for all t in b_idx, counts[t] > 1
                can_remove = True
                for t_idx in block_cover_indices[b_idx]:
                    if counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    to_remove.append(b_idx)
                    
            if to_remove:
                # Remove all identified blocks (they are independent in terms of being removable? 
                # Not necessarily, but usually yes. To be safe, we can remove one by one or all if we re-verify.
                # Given the small size, removing all at once is a good heuristic, but might break validity if they shared critical triples?
                # No, if counts[t] > 1 for all t in b1, and b2 is also removed, counts[t] decreases by 2.
                # If counts[t] was 2, and t is in both b1 and b2, removing both makes it 0.
                # So we must be careful.
                # Let's remove one by one to be safe, or just mark them and re-evaluate.
                # Simpler: Remove one, update counts, restart the scan.
                # For speed, let's try to remove all "safe" ones.
                # A block b is safe if for all t in b, counts[t] > 1.
                # If we remove b1 and b2, and t is in both, counts[t] drops by 2.
                # If counts[t] was 2, it becomes 0. Invalid.
                # So we can only remove a set S if for all t, counts[t] - |{b in S : t in b}| >= 1.
                
                # Let's just remove one at a time to ensure correctness, it's fast enough.
                b_to_remove = to_remove[0]
                for t_idx in block_cover_indices[b_to_remove]:
                    counts[t_idx] -= 1
                current_blocks.remove(b_to_remove)
                improved = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = current_blocks[:]
                continue # Restart the while loop to re-evaluate other blocks
            
            # Phase 2: Try 2-Opt Swaps
            # Find a pair (b1, b2) and a replacement b3 such that
            # (Current - {b1, b2} + {b3}) is valid and size decreases by 1.
            
            n_blocks = len(current_blocks)
            if n_blocks < 2:
                break
                
            # To avoid O(N^2 * M) where M is cost of checking replacement, we can limit checks or optimize.
            # N ~ 80. Pairs ~ 3200.
            # For each pair, identify U_def (triples that would become uncovered).
            # If U_def is empty, we can just remove b1, b2? 
            # If U_def is empty, it means counts[t] >= 2 for t in b1\b2, counts[t] >= 2 for t in b2\b1, counts[t] >= 3 for t in b1^b2.
            # In that case, we can remove both. Size -2.
            
            found_swap = False
            # Shuffle pairs to add randomness?
            pairs = [(i, j) for i in range(n_blocks) for j in range(i+1, n_blocks)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                if (time.time() - start_time) > timeout:
                    break
                    
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                # Calculate U_def: Triples that would have count 0 if b1, b2 removed.
                # t is in U_def if:
                # 1. t is in b1 or b2
                # 2. counts[t] - (1 if t in b1 else 0) - (1 if t in b2 else 0) < 1
                
                u_def = set()
                
                # Iterate through triples in b1
                for t_idx in block_cover_indices[b1]:
                    # Count contribution from b1 and b2
                    # Note: t_idx might be in b2 as well
                    # We need to know if t_idx is in b2.
                    # We can precompute a set of t-indices for each block?
                    # block_cover_indices are lists.
                    # Let's use a set for b2's triples for O(1) lookup?
                    # Creating sets for every pair is expensive.
                    # Alternative:
                    # counts[t] is global.
                    # If t in b1:
                    #   if t in b2: new_count = counts[t] - 2
                    #   else: new_count = counts[t] - 1
                    # We need new_count < 1 => new_count == 0.
                    
                    # To check "t in b2" efficiently, we can precompute a boolean matrix or use sets.
                    # Let's precompute `block_triple_sets` as a list of sets.
                    pass # Placeholder, will implement below
                
                # This logic is getting complex inside the loop. 
                # Let's precompute `block_triple_sets` once.
                # And maybe `block_triple_sets` as frozensets for hashing?
                
                # Actually, let's optimize the U_def calculation.
                # We can just iterate through the union of triples in b1 and b2.
                # Union size <= 35 + 35 = 70.
                
                t_set_b1 = set(block_cover_indices[b1])
                t_set_b2 = set(block_cover_indices[b2])
                union_t = t_set_b1.union(t_set_b2)
                
                for t_idx in union_t:
                    # Calculate remaining count if b1, b2 removed
                    rem_count = counts[t_idx]
                    if t_idx in t_set_b1:
                        rem_count -= 1
                    if t_idx in t_set_b2:
                        rem_count -= 1
                        
                    if rem_count < 1:
                        u_def.add(t_idx)
                        
                if not u_def:
                    # We can remove both b1 and b2
                    for t_idx in t_set_b1:
                        counts[t_idx] -= 1
                    for t_idx in t_set_b2:
                        # If t is in both, we subtracted twice?
                        # No, we subtracted 1 for b1, 1 for b2.
                        # If t in both, counts[t] -= 2. Correct.
                        counts[t_idx] -= 1
                    # Wait, if t in both, the loop above does:
                    # t in b1: counts[t] -= 1
                    # t in b2: counts[t] -= 1
                    # Total -2. Correct.
                    
                    current_blocks.remove(b1)
                    current_blocks.remove(b2)
                    improved = True
                    found_swap = True
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = current_blocks[:]
                    break # Restart main loop to re-evaluate single removals etc.
                else:
                    # Need to find a replacement block b3 that covers all t in u_def
                    # Find intersection of triple_blocks[t] for t in u_def
                    # u_def is small (usually 1-3 elements)
                    
                    if len(u_def) == 0:
                        continue
                        
                    # Start with the block list of the first triple in u_def
                    first_t = next(iter(u_def))
                    candidates = triple_blocks[first_t]
                    
                    # Filter candidates: must cover ALL t in u_def
                    valid_candidates = []
                    for c_idx in candidates:
                        # Check if c_idx covers all t in u_def
                        # c_idx's triples: block_cover_indices[c_idx]
                        # We can check if u_def is subset of block_cover_indices[c_idx]
                        # Using sets is fast for small u_def
                        c_set = set(block_cover_indices[c_idx])
                        if u_def.issubset(c_set):
                            # Also, c_idx should not be b1 or b2 (though if it is, removing b1,b2 and adding c_idx which is b1 is just removing b2? 
                            # If c_idx == b1, then we replace {b1, b2} with {b1}, effectively removing b2. 
                            # This is a single removal, which Phase 1 should have handled if valid.
                            # But maybe b2 was not removable alone because it covered some t only covered by b2?
                            # If c_idx == b1, then b1 covers u_def.
                            # u_def are triples that become uncovered if b1, b2 removed.
                            # If b1 covers them, they are not uncovered if b1 stays.
                            # So replacing {b1, b2} with {b1} is valid if b1 covers u_def.
                            # This is equivalent to removing b2.
                            # Phase 1 checks if b2 is removable.
                            # b2 is removable if all its triples have count > 1.
                            # If u_def is non-empty, some triple t in b2 has count 1 (and is in b1? No, if t in b1 and count 1, then t is only covered by b1. 
                            # Then removing b2 doesn't affect t. 
                            # Wait. If t in u_def, it means count[t] - (1 if t in b1) - (1 if t in b2) < 1.
                            # If t in b1 and t in b2, count[t] must be 2 for t to be in u_def (2-2=0).
                            # If t in b2 only, count[t] must be 1 for t to be in u_def (1-1=0).
                            # If t in b1 only, count[t] must be 1.
                            
                            # If we replace {b1, b2} with {b1}, we keep b1.
                            # The new count for t is counts[t] - 1 (removed b2).
                            # For t in u_def:
                            # Case 1: t in b2 only. count[t]=1. New count 0. INVALID.
                            # So if u_def contains any t in b2 \ b1, we cannot just remove b2.
                            # Thus c_idx must cover those t.
                            # If c_idx == b1, b1 does not cover t (since t in b2 \ b1). So b1 is not a valid candidate.
                            # So we can just accept any valid candidate.
                            
                            valid_candidates.append(c_idx)
                            
                    if valid_candidates:
                        # Pick one (random or first)
                        b3 = random.choice(valid_candidates)
                        
                        # Apply swap: Remove b1, b2, Add b3
                        # Update counts
                        for t_idx in t_set_b1:
                            counts[t_idx] -= 1
                        for t_idx in t_set_b2:
                            counts[t_idx] -= 1
                        for t_idx in block_cover_indices[b3]:
                            counts[t_idx] += 1
                            
                        current_blocks.remove(b1)
                        current_blocks.remove(b2)
                        current_blocks.append(b3)
                        
                        improved = True
                        found_swap = True
                        if len(current_blocks) < best_size:
                            best_size = len(current_blocks)
                            best_blocks = current_blocks[:]
                        break # Restart main loop
                        
            if not found_swap:
                # No single removal or 2-opt swap found
                # We might be stuck. 
                # We can try to perturb: Randomly remove a block and re-add best replacement? 
                # Or just break and return best found.
                # Given the time limit, breaking is safer.
                break
                
        return best_blocks

    # Run Greedy Init
    init_blocks = greedy_init()
    
    # Run Local Search
    # Allow multiple restarts if time permits?
    # Greedy init is fast. Local search is the main cost.
    # Let's do 10 restarts of the whole process (Init + Local Search) if time allows.
    
    best_overall = init_blocks
    best_size_overall = len(init_blocks)
    
    start_total = time.time()
    timeout_total = 115
    
    for restart in range(20):
        if time.time() - start_total > timeout_total:
            break
            
        # Randomize the greedy init by shuffling the order of uncovered selection?
        # The greedy_init uses random.choice for target_t_idx. 
        # So each call is different.
        
        init_b = greedy_init()
        
        # Local search
        # Timeout for local search: leave some time for next restart
        time_left = timeout_total - (time.time() - start_total)
        ls_timeout = max(1, time_left * 0.8) # Use 80% of remaining time for this LS
        
        improved_b = local_search(init_b, ls_timeout)
        
        if len(improved_b) < best_size_overall:
            best_size_overall = len(improved_b)
            best_overall = improved_b
            
    # Convert indices back to blocks
    final_blocks = [all_k_subsets[idx] for idx in best_overall]
    
    # Final validation check (just in case)
    # It's expensive, but ensures correctness.
    # For v=24, k=7, t=3, n_t=2024.
    # We can skip if we trust the logic.
    
    return final_blocks
