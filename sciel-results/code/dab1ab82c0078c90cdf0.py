import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should aim to finish by 140s.
    deadline = start_time + 140.0 

    # 1. Precomputation
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    # Map t-subsets to indices
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # Build sparse incidence matrix M (num_blocks x num_t_subs)
    # M[i, j] = 1 if t_sub j is contained in block i
    rows = []
    cols = []
    
    # Precompute t-subsets for each block to speed up matrix building
    # and for later use in reduction/local search
    block_t_indices = []
    
    for i, block in enumerate(all_blocks):
        block_t_subs = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in block_t_subs]
        block_t_indices.append(indices)
        for j in indices:
            rows.append(i)
            cols.append(j)
            
    data = np.ones(len(rows), dtype=np.int32)
    M = csr_matrix((data, (rows, cols)), shape=(num_blocks, num_t_subs))
    
    # Convert block_t_indices to numpy arrays for faster access
    block_t_indices_np = [np.array(idx, dtype=np.int32) for idx in block_t_indices]
    
    # Precompute sets for faster membership checks in local search
    block_t_sets = [set(idx) for idx in block_t_indices]
    
    # Precompute for each t-subset, which blocks cover it (column indices of M)
    # This will be useful for local search
    M_T = M.T.tocsr()
    
    best_solution = None
    best_len = float('inf')
    
    def greedy_construct(seed_val):
        """
        Vectorized greedy construction with weighted scoring.
        Returns list of selected block indices.
        """
        random.seed(seed_val)
        counts = np.zeros(num_t_subs, dtype=np.int32)
        uncovered = np.ones(num_t_subs, dtype=np.int32)
        selected_blocks = []
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gain for each block: number of uncovered t-subsets it covers
            gains = M.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # All covered
            
            # Find indices with max gain
            best_indices = np.where(gains == max_gain)[0]
            
            # If there are many ties, pick one randomly to diversify
            if len(best_indices) > 1:
                best_idx = best_indices[random.randint(0, len(best_indices) - 1)]
            else:
                best_idx = best_indices[0]
            
            selected_blocks.append(best_idx)
            
            # Update coverage
            idxs = block_t_indices_np[best_idx]
            counts[idxs] += 1
            
            # Check if done
            if np.count_nonzero(uncovered) == 0:
                break
                
            # Vectorized update of uncovered
            uncovered[idxs] = (counts[idxs] == 0)
            
        return selected_blocks, counts

    def reduce_covering(selected_blocks, counts, max_passes=10):
        """
        Remove redundant blocks.
        A block is redundant if all t-subsets it covers have count > 1.
        Multiple passes to catch cascading redundancies.
        """
        if not selected_blocks:
            return selected_blocks, counts
            
        current_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                idxs = block_t_indices_np[b_idx]
                # Check if any of the t-subsets has count == 1
                if np.any(cur_counts[idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Remove it
                    cur_counts[idxs] -= 1
                    changed = True
            current_blocks = new_blocks
            if not changed:
                break
            
        return current_blocks, cur_counts

    def local_search(selected_blocks, counts, seed_val, max_iters=1000):
        """
        Improved local search with swap-based operations.
        Strategy:
        1. Try to remove a block. If it breaks coverage, find a replacement that covers the broken parts.
        2. After a swap, immediately try to remove other blocks that became redundant.
        3. Use the precomputed M_T to efficiently find candidate replacements.
        """
        random.seed(seed_val)
        n_sel = len(selected_blocks)
        if n_sel == 0:
            return selected_blocks, counts

        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        # Track which blocks are currently selected for O(1) lookup
        selected_set = set(sel_blocks)
        
        for iter_num in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random block to try to remove
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices_np[rem_idx]
            
            # Check if removing it leaves any t-subset uncovered
            broken_ts = rem_ts[cur_counts[rem_ts] == 1]
            
            if len(broken_ts) == 0:
                # It doesn't break coverage. Remove it.
                cur_counts[rem_ts] -= 1
                sel_blocks.pop(rem_pos)
                selected_set.discard(rem_idx)
                
                # After removal, try to remove more blocks that became redundant
                # Do a quick reduction pass
                for _ in range(3):
                    if time.time() > deadline - 5.0:
                        break
                    changed = False
                    new_blocks = []
                    for b_idx in sel_blocks:
                        idxs = block_t_indices_np[b_idx]
                        if np.any(cur_counts[idxs] == 1):
                            new_blocks.append(b_idx)
                        else:
                            cur_counts[idxs] -= 1
                            changed = True
                    sel_blocks = new_blocks
                    if not changed:
                        break
            else:
                # It would break coverage. Try to find a replacement.
                # Use M_T to find blocks that cover the broken t-subsets
                # broken_mask is a vector of length num_t_subs with 1s at broken positions
                broken_mask = np.zeros(num_t_subs, dtype=np.int32)
                broken_mask[broken_ts] = 1
                
                # M_T.dot(broken_mask) gives for each block how many broken_ts it covers
                # This is a sparse matrix-vector product
                coverage_counts = M_T.dot(broken_mask)
                
                # Find blocks that cover all broken_ts
                full_cover_mask = coverage_counts == len(broken_ts)
                full_cover_blocks = np.where(full_cover_mask)[0]
                
                # Filter out already-selected blocks
                candidates = [b for b in full_cover_blocks if b not in selected_set]
                
                if candidates:
                    # Pick a random candidate (or the one with best "quality")
                    # For now, pick randomly to diversify
                    best_add = candidates[random.randint(0, len(candidates) - 1)]
                    
                    # Perform swap
                    cur_counts[rem_ts] -= 1
                    add_ts = block_t_indices_np[best_add]
                    cur_counts[add_ts] += 1
                    
                    sel_blocks[rem_pos] = best_add
                    selected_set.discard(rem_idx)
                    selected_set.add(best_add)
                    
                    # After swap, try to remove blocks that became redundant
                    for _ in range(3):
                        if time.time() > deadline - 5.0:
                            break
                        changed = False
                        new_blocks = []
                        for b_idx in sel_blocks:
                            idxs = block_t_indices_np[b_idx]
                            if np.any(cur_counts[idxs] == 1):
                                new_blocks.append(b_idx)
                            else:
                                cur_counts[idxs] -= 1
                                changed = True
                        sel_blocks = new_blocks
                        if not changed:
                            break
                else:
                    # No full cover replacement found. Try partial cover + hope for the best?
                    # Or skip this removal attempt.
                    pass
                
        return sel_blocks, cur_counts

    def perturbation(selected_blocks, counts, seed_val, num_perturb=5):
        """
        Apply random perturbations to escape local optima.
        Remove a few random blocks and repair the covering.
        """
        random.seed(seed_val)
        if not selected_blocks:
            return selected_blocks, counts
        
        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        # Remove a few random blocks
        for _ in range(num_perturb):
            if not sel_blocks:
                break
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices_np[rem_idx]
            cur_counts[rem_ts] -= 1
            sel_blocks.pop(rem_pos)
        
        # Repair: greedily add blocks to cover uncovered t-subsets
        uncovered = (cur_counts == 0).astype(np.int32)
        
        while np.count_nonzero(uncovered) > 0:
            if time.time() > deadline - 5.0:
                break
            gains = M.dot(uncovered)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            best_indices = np.where(gains == max_gain)[0]
            best_idx = best_indices[random.randint(0, len(best_indices) - 1)]
            
            sel_blocks.append(best_idx)
            idxs = block_t_indices_np[best_idx]
            cur_counts[idxs] += 1
            uncovered[idxs] = (cur_counts[idxs] == 0)
        
        return sel_blocks, cur_counts

    # Main Loop: Try multiple seeds for greedy + reduction + local search
    num_restarts = 15  # Adjust based on time
    
    for seed in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # 1. Greedy
        sol, counts = greedy_construct(seed * 12345 + 1)
        
        # 2. Reduction
        sol, counts = reduce_covering(sol, counts)
        
        current_len = len(sol)
        
        # 3. Local Search
        if time.time() < deadline - 25.0:
            sol, counts = local_search(sol, counts, seed * 54321 + 2, max_iters=500)
            current_len = len(sol)
            
        # 4. Optional: Perturbation + Local Search if we have time
        if time.time() < deadline - 40.0 and current_len > best_len - 5:
            sol_pert, counts_pert = perturbation(sol, counts, seed * 98765 + 3, num_perturb=3)
            sol_pert, counts_pert = reduce_covering(sol_pert, counts_pert)
            sol_pert, counts_pert = local_search(sol_pert, counts_pert, seed * 13579 + 4, max_iters=300)
            if len(sol_pert) < current_len:
                sol, counts = sol_pert, counts_pert
                current_len = len(sol)
        
        if current_len < best_len:
            best_len = current_len
            best_solution = sol[:]
            
        # Early termination if we've found a very good solution
        if best_len <= 300:
            break
            
    # Fallback if no solution found (should not happen)
    if best_solution is None:
        best_solution = list(range(num_blocks))
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return final_blocks
