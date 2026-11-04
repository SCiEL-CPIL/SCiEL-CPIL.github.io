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
    
    best_solution = None
    best_len = float('inf')
    
    def greedy_construct(seed_val):
        """
        Vectorized greedy construction.
        Returns list of selected block indices.
        """
        random.seed(seed_val)
        counts = np.zeros(num_t_subs, dtype=np.int32)
        uncovered = np.ones(num_t_subs, dtype=np.int32)
        selected_blocks = []
        
        # We can speed this up by noting that M @ uncovered gives gains.
        # However, M is sparse, so M.dot(uncovered) is fast.
        
        while True:
            if time.time() > deadline - 5.0:
                break
                
            # Calculate gain for each block: number of uncovered t-subsets it covers
            gains = M.dot(uncovered)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # All covered
            
            # Find indices with max gain
            # To add randomness, we can pick randomly among the best ones
            best_indices = np.where(gains == max_gain)[0]
            
            # If there are many ties, pick one randomly to diversify
            if len(best_indices) > 1:
                # Use random seed for reproducibility of this specific run
                best_idx = best_indices[random.randint(0, len(best_indices) - 1)]
            else:
                best_idx = best_indices[0]
            
            selected_blocks.append(best_idx)
            
            # Update coverage
            idxs = block_t_indices_np[best_idx]
            counts[idxs] += 1
            # Update uncovered mask
            # If count becomes 1, it was uncovered (0) and is now covered (1)
            # We need to set uncovered[j] = 0 where counts[j] became 1
            # Actually, simpler:
            # uncovered[j] = 0 if counts[j] > 0
            # We can just update the specific indices
            for j in idxs:
                if counts[j] == 1:
                    uncovered[j] = 0
            
            # Optimization: Instead of loop, we can do:
            # new_uncovered = uncovered.copy()
            # new_uncovered[idxs] = (counts[idxs] == 0)
            # But loop is okay for small k, t.
            # Let's use vectorized update for safety and speed
            uncovered[idxs] = (counts[idxs] == 0)
            
            # Check if done
            if np.count_nonzero(uncovered) == 0:
                break
                
        return selected_blocks, counts

    def reduce_covering(selected_blocks, counts):
        """
        Remove redundant blocks.
        A block is redundant if all t-subsets it covers have count > 1.
        """
        if not selected_blocks:
            return selected_blocks, counts
            
        # We iterate until no more changes
        changed = True
        current_blocks = selected_blocks[:]
        
        while changed:
            if time.time() > deadline - 5.0:
                break
            changed = False
            new_blocks = []
            for b_idx in current_blocks:
                idxs = block_t_indices_np[b_idx]
                # Check if any of the t-subsets has count == 1
                # If yes, the block is essential.
                # If no, it is redundant.
                if np.any(counts[idxs] == 1):
                    new_blocks.append(b_idx)
                else:
                    # Remove it
                    counts[idxs] -= 1
                    changed = True
            current_blocks = new_blocks
            
        return current_blocks, counts

    def local_search(selected_blocks, counts, seed_val, max_iters=500):
        """
        Try to swap blocks to reduce the count further or maintain count with better structure?
        Actually, we just want to minimize count.
        If we can remove a block and add another that covers the same unique parts, count stays same.
        But if we can remove a block and NO other block covers its unique parts, we can't remove it.
        Wait, the reduction phase already removes blocks that don't cover unique parts.
        So after reduction, every block covers at least one unique t-subset.
        To reduce the number of blocks, we need to find a block B1 such that its unique t-subsets can be covered by adding a different block B2, and removing B1.
        But B2 might not be in the set.
        
        A simpler local search:
        1. Identify blocks that cover only 1 unique t-subset.
        2. Try to find another block (in or out of set) that covers that t-subset and potentially more.
        
        This is complex. Let's stick to a simpler heuristic:
        Randomly try to remove a block. If it breaks coverage, try to add a block that fixes the most broken parts.
        """
        random.seed(seed_val)
        n_sel = len(selected_blocks)
        if n_sel == 0:
            return selected_blocks, counts

        # We work with a copy
        sel_blocks = selected_blocks[:]
        cur_counts = counts.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random block to try to remove
            rem_pos = random.randint(0, len(sel_blocks) - 1)
            rem_idx = sel_blocks[rem_pos]
            rem_ts = block_t_indices_np[rem_idx]
            
            # Check if removing it leaves any t-subset uncovered
            # i.e., if any count in rem_ts is 1
            if np.any(cur_counts[rem_ts] == 1):
                # It would break coverage.
                # We can try to swap: remove rem_idx, add a new_idx that covers the broken ones.
                broken_ts = rem_ts[cur_counts[rem_ts] == 1]
                
                # Find a block that covers the most broken_ts
                # This is a set cover problem on a small set.
                # Candidate blocks: all blocks that cover at least one broken_ts.
                # Since broken_ts is small, we can iterate through all blocks?
                # No, 75k blocks.
                # But we can precompute which blocks cover which t-subsets?
                # We have M.
                # Let's just sample some candidates.
                
                best_add = -1
                best_gain = 0
                
                # Sample 100 random blocks
                candidates = random.sample(range(num_blocks), min(100, num_blocks))
                for b_idx in candidates:
                    if b_idx in sel_blocks:
                        continue
                    add_ts = block_t_indices_np[b_idx]
                    # Count how many broken_ts are covered by add_ts
                    # Intersection of broken_ts and add_ts
                    # Since they are sorted arrays, we can use set intersection for speed if small
                    # Or just check membership.
                    # broken_ts is small (usually 1-5 elements)
                    gain = 0
                    for ts in broken_ts:
                        # Check if ts is in add_ts
                        # add_ts is a numpy array. np.isin is slow for single element.
                        # Let's convert to set for this check? Or just loop.
                        if ts in add_ts:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_add = b_idx
                        if gain == len(broken_ts):
                            break # Perfect cover for the broken parts
                
                if best_add != -1 and best_gain == len(broken_ts):
                    # Perform swap
                    # Remove rem_idx
                    cur_counts[rem_ts] -= 1
                    # Add best_add
                    add_ts = block_t_indices_np[best_add]
                    cur_counts[add_ts] += 1
                    
                    # Update list
                    sel_blocks[rem_pos] = best_add
                    # Note: This doesn't reduce the number of blocks, just changes them.
                    # It might enable further reductions?
                    # Let's run a quick reduction pass after a swap?
                    # Too expensive.
                    
            else:
                # It doesn't break coverage.
                # Why wasn't it removed in the reduction phase?
                # Maybe it was essential then, but due to other changes, it's not now?
                # Or maybe the reduction phase didn't catch it because it depends on order?
                # Let's remove it and update counts.
                cur_counts[rem_ts] -= 1
                sel_blocks.pop(rem_pos)
                # Since we removed a block, we might have made other blocks redundant.
                # We should run reduction again?
                # For now, just continue. The next iteration might catch redundancies.
                
        return sel_blocks, cur_counts

    # Main Loop: Try multiple seeds for greedy + reduction + local search
    # The greedy phase is deterministic if we don't randomize ties, but we do.
    # So different seeds give different coverings.
    
    num_restarts = 20 # Adjust based on time
    
    for seed in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Greedy
        sol, counts = greedy_construct(seed * 12345 + 1)
        
        # 2. Reduction
        sol, counts = reduce_covering(sol, counts)
        
        current_len = len(sol)
        
        # 3. Local Search
        # Only do local search if we have time
        if time.time() < deadline - 20.0:
            sol, counts = local_search(sol, counts, seed * 54321 + 2, max_iters=200)
            current_len = len(sol)
            
        if current_len < best_len:
            best_len = current_len
            best_solution = sol[:]
            
    # Fallback if no solution found (should not happen)
    if best_solution is None:
        # Run one greedy without time pressure if possible?
        # Or just return a trivial valid covering (all blocks)
        best_solution = list(range(num_blocks))
        
    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return final_blocks
