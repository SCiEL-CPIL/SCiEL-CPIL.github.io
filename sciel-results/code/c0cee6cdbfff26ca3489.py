import numpy as np
from scipy.sparse import csr_matrix
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    # Map t-subsets to indices
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Build incidence matrix M (num_blocks x num_t_subs)
    rows = []
    cols = []
    for i, block in enumerate(all_blocks):
        block_t_subs = itertools.combinations(block, t)
        for ts in block_t_subs:
            j = t_sub_index[ts]
            rows.append(i)
            cols.append(j)
            
    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(num_blocks, num_t_subs))
    
    # Convert to dense for faster random access during local search
    # M_dense[i, j] = 1 if t-subset j is in block i
    M_dense = M.toarray().astype(np.int8)
    
    # Precompute block t-subset indices for fast redundancy checks
    block_t_indices = []
    for i, block in enumerate(all_blocks):
        indices = [t_sub_index[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)
    
    def greedy_construct(randomness=0.0):
        """
        Greedy construction with optional randomization (GRASP candidate list).
        randomness in [0, 1]: 0 = deterministic, 1 = fully random among top candidates
        """
        counts = np.zeros(num_t_subs, dtype=np.int32)
        uncovered_mask = np.ones(num_t_subs, dtype=np.int32)
        selected_blocks = []
        selected_set = set()
        
        while True:
            if time.time() > deadline - 5.0:
                break
            
            # Calculate gain for each unselected block
            gains = M_dense @ uncovered_mask
            
            # Mask out already selected blocks
            for b in selected_blocks:
                gains[b] = -1
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if randomness == 0.0:
                best_idx = np.argmax(gains)
            else:
                # GRASP: select from top candidates
                # Find all blocks with gain close to max
                threshold = max_gain - int(randomness * max_gain * 0.5)
                candidates = np.where(gains >= threshold)[0]
                if len(candidates) == 0:
                    best_idx = np.argmax(gains)
                else:
                    # Weighted random selection favoring higher gains
                    weights = gains[candidates].astype(np.float64)
                    weights = np.maximum(weights, 1e-10)
                    weights /= np.sum(weights)
                    best_idx = candidates[np.random.choice(len(candidates), p=weights)]
            
            selected_blocks.append(int(best_idx))
            selected_set.add(int(best_idx))
            
            # Update counts and uncovered mask
            row = M_dense[best_idx]
            counts += row
            uncovered_mask = (counts == 0).astype(np.int32)
            
            if not np.any(uncovered_mask):
                break
        
        return selected_blocks
    
    def local_search(selected_blocks):
        """
        Local search: 1-exchange (remove redundant, then try 1-for-1 swaps).
        Returns improved selected_blocks.
        """
        selected_set = set(selected_blocks)
        
        # Build counts for current solution
        counts = np.zeros(num_t_subs, dtype=np.int32)
        for b in selected_blocks:
            counts += M_dense[b]
        
        improved = True
        while improved:
            if time.time() > deadline - 3.0:
                break
            improved = False
            
            # Phase 1: Remove redundant blocks
            new_selected = []
            for b_idx in selected_blocks:
                if time.time() > deadline - 3.0:
                    new_selected.append(b_idx)
                    continue
                
                # Check if block b_idx is redundant
                redundant = True
                for ts_idx in block_t_indices[b_idx]:
                    if counts[ts_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    counts -= M_dense[b_idx]
                    selected_set.discard(b_idx)
                    improved = True
                else:
                    new_selected.append(b_idx)
            
            selected_blocks = new_selected
            
            # Phase 2: Try 1-for-1 exchanges
            if time.time() > deadline - 4.0:
                break
                
            for i, b_out in enumerate(list(selected_blocks)):
                if time.time() > deadline - 4.0:
                    break
                
                # Find best replacement
                # Gain of adding b_in and removing b_out
                # Net gain = (new t-subs covered by b_in not covered by others after removing b_out) 
                #           - (t-subs only covered by b_out)
                
                # First, check if b_out can be removed (i.e., it's redundant given others)
                # But we want to swap, so we need b_in to cover what b_out uniquely covers
                
                # Compute what b_out uniquely covers
                unique_to_bout = []
                for ts_idx in block_t_indices[b_out]:
                    if counts[ts_idx] == 1:
                        unique_to_bout.append(ts_idx)
                
                if not unique_to_bout:
                    # b_out is redundant, just remove it (already handled in Phase 1)
                    continue
                
                # Find candidate b_in that covers all unique_to_bout
                # and has high net gain
                # For efficiency, only check blocks that cover at least one unique t-subset
                
                # Get all t-subsets covered by b_out
                bout_tset = set(block_t_indices[b_out])
                
                best_in = -1
                best_net_gain = 0
                
                # Iterate over potential replacements
                # To speed up, we can limit to blocks that share many t-subsets with b_out
                # But for now, let's do a bounded search
                
                # Compute current coverage without b_out
                # counts_without_bout = counts - M_dense[b_out]
                # For each candidate b_in, net gain = |b_in ∩ uncovered_after_removing_bout| 
                #                                    - |b_out_unique_not_in_b_in|
                
                # Simplified: just try a limited number of candidates
                # Candidates: blocks that cover at least one of the unique t-subsets of b_out
                
                # Build a quick index: for each t-subset, which blocks cover it?
                # This is expensive to build, so let's just sample
                
                # Alternative: try all blocks but with early termination
                # Actually, let's do a simpler approach: try removing b_out and 
                # greedily adding the best single block that covers the most newly-uncovered t-subs
                
                # Temporarily remove b_out
                counts -= M_dense[b_out]
                newly_uncovered = (counts == 0)
                
                # Find best block to add
                gains = M_dense @ newly_uncovered
                # Exclude already selected blocks
                for b in selected_blocks:
                    if b != b_out:
                        gains[b] = -1
                
                max_gain = np.max(gains)
                
                if max_gain > 0:
                    best_in = int(np.argmax(gains))
                    # Check if this covers all unique t-subsets
                    # The unique t-subsets are those that were count==1 before removal
                    # After removal, they have count==0, so they're in newly_uncovered
                    # If best_in covers all of them, it's a valid swap
                    
                    # Actually, we need to verify: does best_in cover all unique_to_bout?
                    # unique_to_bout are the t-subs with count==1 before removal
                    # After removal, they have count==0
                    # So we need best_in to cover all of them
                    
                    # Check
                    all_covered = True
                    for ts_idx in unique_to_bout:
                        if M_dense[best_in, ts_idx] == 0:
                            all_covered = False
                            break
                    
                    if all_covered:
                        # Valid swap
                        counts += M_dense[best_in]
                        selected_blocks[i] = best_in
                        selected_set.discard(b_out)
                        selected_set.add(best_in)
                        improved = True
                    else:
                        # Not a valid swap, restore
                        counts += M_dense[b_out]
                else:
                    # No good replacement, restore
                    counts += M_dense[b_out]
        
        return selected_blocks
    
    # GRASP: Multiple restarts with randomized construction
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 0
    while time.time() < deadline - 10.0:
        num_restarts += 1
        
        # Increase randomness over time to explore more
        base_randomness = min(0.3 + 0.02 * num_restarts, 0.8)
        randomness = base_randomness * random.random()
        
        # Construct initial solution
        sol = greedy_construct(randomness)
        
        # Local search
        sol = local_search(sol)
        
        # Update best
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol[:]
        
        # If we've found a very good solution, reduce exploration
        if best_size <= 340:
            break
    
    # Final local search on best solution with more time
    if best_solution is not None and time.time() < deadline - 5.0:
        best_solution = local_search(best_solution)
    
    # Convert indices back to blocks
    final_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return final_blocks
