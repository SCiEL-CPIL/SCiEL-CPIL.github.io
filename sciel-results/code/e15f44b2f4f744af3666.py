import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = list(range(v))
    
    # Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute for each block, the indices of t-subsets it covers
    block_cover_indices = []
    for b, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
    
    # Build inverse: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, indices in enumerate(block_cover_indices):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for fast indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Helper: Greedy Construction with sparse gain updates
    def greedy_construction(rng, randomized=False, top_k=10):
        uncovered = np.ones(n_t_subsets, dtype=bool)
        # Initial gains: all blocks cover C(k,t) subsets
        k_choose_t = len(block_cover_indices[0])
        gains = np.full(n_k_blocks, k_choose_t, dtype=np.int32)
        
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            if randomized:
                # Pick from top-k candidates with positive gain
                positive_mask = gains > 0
                if not np.any(positive_mask):
                    break
                valid_blocks = np.where(positive_mask)[0]
                if len(valid_blocks) > top_k:
                    # Get top-k by gain
                    valid_gains = gains[valid_blocks]
                    top_idx = np.argpartition(valid_gains, -top_k)[-top_k:]
                    top_blocks = valid_blocks[top_idx]
                    # Weighted random selection based on gain
                    weights = gains[top_blocks].astype(float)
                    weights /= weights.sum()
                    best_block = top_blocks[rng.choice(len(top_blocks), p=weights)]
                else:
                    best_block = valid_blocks[rng.integers(0, len(valid_blocks))]
            else:
                # Pure greedy: pick max gain
                best_block = np.argmax(gains)
                if gains[best_block] == 0:
                    break
            
            selected_indices.append(best_block)
            
            # Update: for each t-subset covered by best_block, if it was uncovered,
            # mark as covered and decrement gains for all blocks covering it
            for s_idx in block_cover_indices[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
        
        return selected_indices
    
    # 3. Helper: Local Search with redundancy removal + swap moves
    def local_search(selected_indices, max_rounds=20):
        if not selected_indices:
            return selected_indices
        
        selected_set = set(selected_indices)
        
        # Compute cover counts
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_cover_indices[b_idx]] += 1
        
        improved = True
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            # Phase A: Redundancy removal
            new_selected = []
            for b_idx in selected_indices:
                subs = block_cover_indices[b_idx]
                if np.min(counts[subs]) > 1:
                    # Redundant: remove it
                    counts[subs] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            selected_indices = new_selected
            selected_set = set(selected_indices)
            
            # Phase B: Swap moves - try to replace one block with another
            # For each block, find its "unique" t-subsets (covered only by this block)
            # Then try to find a replacement that covers all unique subsets
            for i, b_idx in enumerate(selected_indices):
                if time.time() > deadline:
                    break
                
                subs = block_cover_indices[b_idx]
                # Find unique subsets: those with count == 1
                unique_mask = counts[subs] == 1
                unique_subs = subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # This block is redundant (should have been caught in Phase A)
                    continue
                
                # Try to find a replacement block that covers all unique_subs
                # Candidate blocks must cover the first unique subset
                first_s = unique_subs[0]
                candidates = tsub_to_blocks_np[first_s]
                
                best_replacement = None
                best_gain = 0  # We want to maximize the number of additional subsets covered
                
                for c in candidates:
                    if c in selected_set:
                        continue
                    
                    # Check if c covers all unique_subs
                    covers_all = True
                    for s in unique_subs[1:]:
                        # Check if c covers s: s must be in block_cover_indices[c]
                        # Since block_cover_indices[c] is a numpy array, use searchsorted or isin
                        if s not in block_cover_indices[c]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Calculate gain: how many NEW subsets does c cover that are not covered by b_idx?
                    # Actually, we want to minimize the total. If we remove b_idx and add c,
                    # the change in count is: -1 for subsets in b_idx not in c, +1 for subsets in c not in b_idx.
                    # But we must ensure no subset drops to 0.
                    # Since c covers all unique_subs, those won't drop to 0.
                    # For other subsets in b_idx, they have count >= 2, so removing b_idx is safe.
                    # For subsets in c but not in b_idx, they gain +1, which is fine.
                    # So any valid replacement works. To be smarter, we could prefer replacements
                    # that cover many subsets already covered by other blocks (to create more redundancy).
                    
                    # For now, just take the first valid replacement
                    best_replacement = c
                    break
                
                if best_replacement is not None:
                    # Perform the swap
                    # Remove b_idx
                    counts[block_cover_indices[b_idx]] -= 1
                    # Add best_replacement
                    counts[block_cover_indices[best_replacement]] += 1
                    
                    # Update selected_indices
                    selected_indices[i] = best_replacement
                    selected_set.discard(b_idx)
                    selected_set.add(best_replacement)
                    improved = True
        
        return selected_indices
    
    # 4. Main Search Loop
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Alternate between pure greedy and randomized greedy
        if iteration % 3 == 0:
            # Randomized greedy with larger top_k for more diversity
            selected_indices = greedy_construction(rng, randomized=True, top_k=20)
        elif iteration % 3 == 1:
            # Pure greedy
            selected_indices = greedy_construction(rng, randomized=False)
        else:
            # Randomized greedy with smaller top_k
            selected_indices = greedy_construction(rng, randomized=True, top_k=5)
        
        if not selected_indices:
            continue
        
        # Apply Local Search
        refined_indices = local_search(selected_indices, max_rounds=10)
        
        # Verify validity
        is_valid = True
        if refined_indices:
            check_counts = np.zeros(n_t_subsets, dtype=np.int32)
            for b_idx in refined_indices:
                check_counts[block_cover_indices[b_idx]] += 1
            if np.any(check_counts == 0):
                is_valid = False
        else:
            is_valid = (n_t_subsets == 0)
        
        if is_valid and len(refined_indices) < best_len:
            best_len = len(refined_indices)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
            
            # If we've reached a good solution, do a more intensive polish
            if best_len <= 160 and time.time() < deadline - 10:
                # Run more rounds of local search on the best solution
                polished = local_search(refined_indices, max_rounds=50)
                if polished:
                    check_counts2 = np.zeros(n_t_subsets, dtype=np.int32)
                    for b_idx in polished:
                        check_counts2[block_cover_indices[b_idx]] += 1
                    if not np.any(check_counts2 == 0) and len(polished) < best_len:
                        best_len = len(polished)
                        best_blocks = [list(k_blocks[i]) for i in polished]
    
    # Fallback
    if best_blocks is None:
        selected_indices = greedy_construction(rng, randomized=False)
        if selected_indices:
            refined_indices = local_search(selected_indices, max_rounds=10)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
        else:
            best_blocks = []
    
    return best_blocks
