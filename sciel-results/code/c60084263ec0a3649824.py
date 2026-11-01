import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s, but the prompt says 120s limit.
    # We will use 115s to be safe.
    deadline = start_time + 115.0
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute block_cover_indices: for each block, indices of t-subsets it covers
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(np.array(indices, dtype=np.int32))
        
    # Precompute tsub_to_blocks: for each t-subset, indices of blocks that cover it
    # This is crucial for efficient gain updates
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, inds in enumerate(block_cover_indices):
        for s_idx in inds:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Convert to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gain for each block is C(k, t)
    k_choose_t = len(block_cover_indices[0])
    
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts to escape local minima
    # We will run greedy + local search multiple times if time permits
    while time.time() < deadline:
        
        # --- Phase 1: Greedy Construction ---
        # gains[b] = number of currently uncovered t-subsets covered by block b
        gains = np.full(n_k_blocks, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_indices = []
        
        # Optimization: Keep track of uncovered count to stop early if 0
        uncovered_count = n_t_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
                
            # Find block with max gain
            # Note: np.argmax is O(N), which is fine for 50k blocks
            b = np.argmax(gains)
            
            if gains[b] == 0:
                # No block covers any uncovered subset? 
                # This shouldn't happen if a cover exists, but break to avoid infinite loop
                break
                
            selected_indices.append(b)
            
            # Update gains and uncovered status
            # Only update for t-subsets that were actually uncovered
            for s in block_cover_indices[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    # Decrement gain for all blocks covering this t-subset
                    # This is the key optimization: O(num_blocks_covering_s) instead of O(N_blocks)
                    gains[tsub_to_blocks_np[s]] -= 1
                    
        # If we didn't finish greedy due to time, we might have an invalid cover
        # We should check validity later, or just skip this iteration if incomplete
        if uncovered_count > 0:
            continue
            
        # --- Phase 2: Local Search ---
        # Compute cover counts
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        for b in selected_indices:
            cover_count[block_cover_indices[b]] += 1
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step 2a: Redundancy Removal
            # Check if any block is redundant
            # A block is redundant if all t-subsets it covers have count > 1
            redundant_blocks = []
            for b in selected_indices:
                # If min count of covered subsets > 1, it's redundant
                if np.min(cover_count[block_cover_indices[b]]) > 1:
                    redundant_blocks.append(b)
                    
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_indices.remove(b)
                    cover_count[block_cover_indices[b]] -= 1
                improved = True
                # Restart local search loop to check for new redundancies
                continue
                
            # Step 2b: 1-for-1 Swaps
            # Calculate unique counts for each block
            # unique_count[b] = number of t-subsets covered only by b
            unique_counts = {}
            unique_tsubs = {} # block -> list of t-subsets unique to it
            
            for b in selected_indices:
                u_count = 0
                u_list = []
                subs = block_cover_indices[b]
                for s in subs:
                    if cover_count[s] == 1:
                        u_count += 1
                        u_list.append(s)
                unique_counts[b] = u_count
                unique_tsubs[b] = u_list
                
            # Find block with minimum unique count (but > 0, as 0 would be redundant and removed)
            # If all are 0, it's a perfect cover with no redundancy, but we already checked redundancy.
            # Actually, if unique_count is 0, it means it's redundant. So min should be >= 1.
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            
            # Candidates for removal are those with min_unique
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            
            # Shuffle to break symmetry
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs[b_out]
                
                # Find a block b_in that covers all s in s_uniq
                # Exclude blocks already in selected_block_indices
                exclude_set = set(selected_indices)
                
                # Candidate blocks must cover the first unique subset
                if not s_uniq:
                    # Should not happen if min_unique > 0
                    continue
                    
                first_s = s_uniq[0]
                candidates_in = tsub_to_blocks_np[first_s]
                
                valid_candidates = []
                for c in candidates_in:
                    if c in exclude_set:
                        continue
                    # Check if c covers all other tsubs in s_uniq
                    covers_all = True
                    for s in s_uniq[1:]:
                        # Check if c is in tsub_to_blocks[s]
                        # Using binary search or set lookup would be faster, but list scan is okay for small lists
                        # tsub_to_blocks[s] is a list of ints
                        if c not in tsub_to_blocks[s]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c)
                        
                if valid_candidates:
                    # Pick a random candidate to explore different solutions
                    b_in = random.choice(valid_candidates)
                    
                    # Perform swap
                    selected_indices.remove(b_out)
                    selected_indices.append(b_in)
                    
                    # Update cover counts
                    cover_count[block_cover_indices[b_out]] -= 1
                    cover_count[block_cover_indices[b_in]] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap to recalculate unique counts
            
            if not swap_made:
                # No swap possible, we are in a local minimum
                break
                
        # End Local Search
        
        # Update best
        current_len = len(selected_indices)
        if current_len < best_len:
            best_len = current_len
            best_blocks = [list(k_blocks[i]) for i in selected_indices]
            
        # Check time for next restart
        if time.time() > deadline:
            break
            
    # Fallback if no valid cover was found (should not happen)
    if best_blocks is None:
        # Try a simple greedy without local search, just to return *something*
        # But likely the first iteration succeeded.
        # If not, we return an empty list which is invalid.
        # Let's just return the last selected_indices if it covers everything.
        # For safety, if best_blocks is None, we try one more quick greedy.
        gains = np.full(n_k_blocks, k_choose_t, dtype=np.int32)
        uncovered = np.ones(n_t_subsets, dtype=bool)
        selected_indices = []
        uncovered_count = n_t_subsets
        while uncovered_count > 0 and time.time() < deadline:
            b = np.argmax(gains)
            if gains[b] == 0: break
            selected_indices.append(b)
            for s in block_cover_indices[b]:
                if uncovered[s]:
                    uncovered[s] = False
                    uncovered_count -= 1
                    gains[tsub_to_blocks_np[s]] -= 1
        best_blocks = [list(k_blocks[i]) for i in selected_indices]

    return best_blocks
