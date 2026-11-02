import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Helper function for combinations count
    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence structures
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)
        
    # Convert ts_to_blocks to numpy arrays for faster intersection if needed, 
    # but lists are fine for small lists (avg ~3000 blocks per ts is too big? 
    # C(19,8)=75582. C(19,5)=116280.
    # Each ts is in C(19-5, 8-5) = C(14,3) = 364 blocks.
    # So ts_to_blocks lists are small (size 364). This is very good for targeted greedy.
    
    # State for Greedy
    # We will use a "Targeted Greedy" approach:
    # 1. Pick an uncovered t-subset.
    # 2. Look at all blocks covering it (only 364 of them).
    # 3. Pick the one that covers the most OTHER uncovered t-subsets.
    
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    
    solution_blocks = []
    
    # Precompute list of all t-subset indices for fast iteration
    all_ts_indices = np.arange(num_t_subsets)
    
    # Targeted Greedy Phase
    while True:
        if time.time() > deadline - 10.0:
            break
            
        # Find any uncovered t-subset
        # Using np.where is fast enough for 116k elements
        uncovered_indices = np.where(is_uncovered)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered t-subset to target
        # Or pick one with fewest covering blocks? 
        # Since all ts have exactly 364 covering blocks initially, random is fine.
        target_ts_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        
        # Get all blocks that cover this target t-subset
        candidate_block_indices = ts_to_blocks[target_ts_idx]
        
        if not candidate_block_indices:
            # Should not happen if valid parameters
            break
            
        best_block_idx = -1
        best_gain = -1
        
        # Evaluate gain for each candidate block
        # Gain = number of currently uncovered t-subsets covered by this block
        for b_idx in candidate_block_indices:
            if is_selected[b_idx]:
                continue
                
            # Calculate gain: count uncovered ts in block_to_ts[b_idx]
            # block_to_ts[b_idx] is a list of ts indices
            # We need to count how many of these are in is_uncovered
            # This is a loop over ~56 elements. Very fast.
            gain = 0
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1:
            # All candidates already selected? Should not happen if we pick from uncovered
            break
            
        # Select the best block
        solution_blocks.append(list(all_k_blocks[best_block_idx]))
        is_selected[best_block_idx] = True
        
        # Update coverage
        for ts_idx in block_to_ts[best_block_idx]:
            cover_count[ts_idx] += 1
            if cover_count[ts_idx] == 1:
                is_uncovered[ts_idx] = False
                
    best_solution = solution_blocks
    best_len = len(solution_blocks)
    
    # Local Search: Remove redundant blocks
    def optimize_redundancy(blocks):
        # Recalculate cover counts from scratch for safety
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            # Find index of block b in all_k_blocks
            # Since blocks are lists of ints, we can convert to tuple and lookup
            # But we stored indices in solution_blocks? No, we stored lists.
            # Let's keep track of indices in the solution for efficiency in local search.
            pass
        
        # This function is called on `blocks` which are lists.
        # We need to map them back to indices.
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        
        # Rebuild cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            idx = block_tuple_to_idx.get(tuple(b), -1)
            if idx >= 0:
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
            
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 1.0:
                    return blocks
                
                idx = block_tuple_to_idx.get(tuple(blocks[i]), -1)
                if idx < 0:
                    continue
                    
                redundant = True
                for ts_idx in block_to_ts[idx]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_to_ts[idx]:
                        cover_count[ts_idx] -= 1
                    changed = True
                    break
                    
        return blocks

    # Local Search: 1-move swap optimization
    def local_search_swap(blocks):
        block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}
        
        # Rebuild cover count and solution indices
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        sol_block_indices = set()
        for b in blocks:
            idx = block_tuple_to_idx.get(tuple(b), -1)
            if idx >= 0:
                sol_block_indices.add(idx)
                for ts_idx in block_to_ts[idx]:
                    cover_count[ts_idx] += 1
                    
        improved = False
        # Randomize order of blocks to try
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 1.0:
                return blocks, improved
                
            idx = block_tuple_to_idx.get(tuple(blocks[i]), -1)
            if idx < 0:
                continue
                
            # Find t-subsets uniquely covered by blocks[i]
            unique_ts = []
            for ts_idx in block_to_ts[idx]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)
                    
            if not unique_ts:
                continue
            
            # Find candidate blocks that cover ALL unique_ts
            # Start with blocks covering the first unique ts
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Remove current block
            candidates.discard(idx)
            
            # Filter out blocks already in solution
            valid_candidates = [c for c in candidates if c not in sol_block_indices]
            
            if not valid_candidates:
                continue
                
            # Pick one randomly (or the one with max gain? Random is faster)
            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            
            # Perform swap
            old_block = blocks[i]
            blocks[i] = new_block
            
            # Update cover counts
            for ts_idx in block_to_ts[idx]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx]:
                cover_count[ts_idx] += 1
            
            # Update solution block indices
            sol_block_indices.discard(idx)
            sol_block_indices.add(new_block_idx)
            
            improved = True
            
        return blocks, improved

    # Run local search for a few iterations
    # First, clean up redundancy
    best_solution = optimize_redundancy(best_solution)
    
    for _ in range(100): # Increase iterations since targeted greedy is faster
        if time.time() > deadline - 2.0:
            break
        best_solution, improved = local_search_swap(best_solution)
        if not improved:
            break
        best_solution = optimize_redundancy(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Final redundancy cleanup
    best_solution = optimize_redundancy(best_solution)

    return best_solution
