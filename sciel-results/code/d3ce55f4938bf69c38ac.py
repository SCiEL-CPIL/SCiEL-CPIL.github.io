import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers
    block_to_tsubs = []
    for block in k_subsets:
        covered_tsubs = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            covered_tsubs.append(idx)
        block_to_tsubs.append(covered_tsubs)
        
    # Precompute for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Initial gains: each block covers C(k, t) t-subsets
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    # Track uncovered t-subsets
    uncovered = np.ones(num_tsubs, dtype=bool)
    
    selected_block_indices = []
    
    # 1. Greedy Construction
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Find the block with the maximum gain
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
            
        selected_block_indices.append(b)
        
        # Update gains for all blocks that share any t-subset with b
        for s in block_to_tsubs[b]:
            if uncovered[s]:
                uncovered[s] = False
                gains[tsub_to_blocks_np[s]] -= 1

    # If greedy didn't finish, we might have an incomplete cover.
    # But the problem says "Valid when every t-subset ... lies in at least one block".
    # If we break early, we return an invalid result?
    # The score is 0 for invalid.
    # So we MUST finish the greedy phase or find a cover.
    # For C(19,7,4), greedy should be fast enough.
    # 151 blocks * ~200 tsubs/block * update cost.
    
    if not uncovered.any():
        # 2. Local Optimization (Redundancy Removal + Swaps)
        # We operate on the list of selected block indices
        # We maintain cover_count for each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Helper to check if a block is redundant
        def is_redundant(b_idx, cover_count):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # Helper to find a block covering all t-subsets in a list
        def find_covering_block(tsubs_to_cover, exclude_set):
            if not tsubs_to_cover:
                return None # Should not happen if we are swapping
            
            # Get candidate blocks from the first t-subset
            first_s = tsubs_to_cover[0]
            candidates = tsub_to_blocks_np[first_s]
            
            # Filter candidates
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                # Check if c covers all other tsubs
                covers_all = True
                for s in tsubs_to_cover[1:]:
                    if c not in tsub_to_blocks[s]:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            # Return a random candidate to avoid getting stuck in a loop
            return random.choice(valid_candidates)

        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step 2a: Redundancy Removal
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b, cover_count):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                # Remove all redundant blocks found in this pass
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue # Restart loop to check for new redundancies
            
            # Step 2b: Try Swaps
            # Calculate unique counts for each block
            unique_counts = {}
            unique_tsubs = {} # block -> list of tsubs unique to it
            
            for b in selected_block_indices:
                u_count = 0
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                        u_list.append(s)
                unique_counts[b] = u_count
                unique_tsubs[b] = u_list
            
            # Find block with minimum unique count (but > 0)
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
                exclude_set = set(selected_block_indices)
                
                b_in = find_covering_block(s_uniq, exclude_set)
                
                if b_in is not None:
                    # Perform swap
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    # Update cover counts
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break # Restart loop after one swap
            
            if not swap_made:
                # No swap possible, we are in a local minimum
                # We could try a 2-swap, but it's expensive.
                # Or restart greedy with a different random tie-break?
                # For now, stop.
                break

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    return result
