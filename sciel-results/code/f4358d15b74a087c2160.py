import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Use a safe deadline to ensure we return a valid result
    deadline = start_time + 115.0 

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if t-subset j is in block i
    # Using uint8 for memory efficiency and fast bitwise ops
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        # Generate all t-subsets within block B
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
    
    # Precompute inverted index: for each t-subset, which blocks cover it
    # This is critical for fast swap candidate lookup
    inverted_index = []
    for j in range(n_tsubs):
        col = Cover[:, j]
        idxs = np.where(col == 1)[0]
        inverted_index.append(idxs)
    inverted_index = np.array(inverted_index, dtype=object)

    # Phase 1: Greedy Set Cover Initialization with multiple restarts
    def greedy_init(randomize=False, max_iter=10000):
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        selected = []
        
        for _ in range(max_iter):
            if np.sum(uncovered) == 0:
                break
            if time.time() > deadline - 40.0:
                break
                
            # Compute gains: number of uncovered t-subsets covered by each block
            gains = Cover @ uncovered
            
            if randomize:
                # Add small random perturbation to break ties and explore
                gains = gains + random.random() * 0.1
                # Only consider blocks with gain > 0
                gains[gains < 1] = 0
            
            best_idx = np.argmax(gains)
            best_gain = gains[best_idx]
            
            if best_gain == 0:
                break
            
            selected.append(best_idx)
            uncovered = uncovered & (Cover[best_idx] == 0)
        
        return selected, uncovered
    
    # Run greedy with a few randomized restarts to find a better starting point
    best_selected = None
    best_uncovered = None
    best_count = float('inf')
    
    n_restarts = 5
    for r in range(n_restarts):
        random.seed(42 + r)
        sel, unc = greedy_init(randomize=(r > 0))
        count = len(sel)
        if np.sum(unc) == 0:  # Valid cover
            if count < best_count:
                best_count = count
                best_selected = sel
                best_uncovered = unc
    
    if best_selected is None:
        # Fallback: deterministic greedy
        best_selected, best_uncovered = greedy_init(randomize=False)
    
    selected_indices = best_selected

    # Phase 2: Redundancy Removal
    def remove_redundants(sel_indices, cover_counts):
        """
        Iteratively remove blocks that are not essential.
        A block is redundant if it doesn't cover any t-subset with count 1.
        """
        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            new_sel_indices = []
            for idx in sel_indices:
                # Check if this block covers any t-subset with count 1
                # Essential if any (Cover[idx, j] == 1 and cover_counts[j] == 1)
                if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                    new_sel_indices.append(idx)
                else:
                    # Remove block
                    cover_counts -= Cover[idx]
                    changed = True
            
            sel_indices = new_sel_indices
        return sel_indices, cover_counts

    # Initialize counts for current selection
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_indices:
        cover_counts += Cover[idx]
        
    selected_indices, cover_counts = remove_redundants(selected_indices, cover_counts)

    # Phase 3: Local Search with 1-for-1 Swaps using precomputed inverted index
    max_swap_iterations = 100
    for _ in range(max_swap_iterations):
        if time.time() > deadline - 10.0:
            break
            
        improved = False
        # Shuffle to avoid bias
        indices_to_try = selected_indices[:]
        random.shuffle(indices_to_try)
        
        for i_idx in indices_to_try:
            if time.time() > deadline - 10.0:
                break
                
            # Find critical t-subsets for block i_idx
            # These are t-subsets covered by i_idx and no other selected block
            # i.e., Cover[i_idx, j] == 1 and cover_counts[j] == 1
            critical_mask = (Cover[i_idx] == 1) & (cover_counts == 1)
            critical_count = np.sum(critical_mask)
            
            if critical_count == 0:
                # Block is redundant
                cover_counts -= Cover[i_idx]
                selected_indices.remove(i_idx)
                improved = True
                break # Restart loop to re-evaluate all
            
            # Find a candidate block j that covers all critical t-subsets
            # Use the precomputed inverted index for speed
            c_idx = np.where(critical_mask)[0][0]
            candidate_indices = inverted_index[c_idx]
            
            # Filter candidates:
            # 1. Not in selected_indices
            # 2. Covers ALL critical t-subsets
            sel_set = set(selected_indices)
            
            found_swap = False
            for j_idx in candidate_indices:
                if j_idx in sel_set:
                    continue
                
                # Check if j_idx covers all critical t-subsets
                if np.sum(Cover[j_idx] & critical_mask) == critical_count:
                    # Valid swap found!
                    cover_counts -= Cover[i_idx]
                    cover_counts += Cover[j_idx]
                    
                    selected_indices.remove(i_idx)
                    selected_indices.append(j_idx)
                    
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                selected_indices, cover_counts = remove_redundants(selected_indices, cover_counts)
                break
    
    # Phase 4: 2-for-1 Exchange Local Search
    # Try to replace two blocks with one block that covers all their critical t-subsets
    # This can reduce the number of blocks by 1 per successful exchange
    max_2for1_iterations = 30
    for _ in range(max_2for1_iterations):
        if time.time() > deadline - 8.0:
            break
            
        improved = False
        # Try pairs of blocks
        n_sel = len(selected_indices)
        # Limit pairs to try to keep time manageable
        max_pairs_to_try = 500
        pair_attempts = 0
        
        for a_idx in range(n_sel):
            if time.time() > deadline - 8.0 or pair_attempts >= max_pairs_to_try:
                break
            for b_idx in range(a_idx + 1, n_sel):
                if time.time() > deadline - 8.0 or pair_attempts >= max_pairs_to_try:
                    break
                pair_attempts += 1
                
                i_idx = selected_indices[a_idx]
                j_idx = selected_indices[b_idx]
                
                # Find t-subsets that are covered ONLY by i_idx or j_idx (among selected)
                # Critical for the pair: t-subsets where count == 1 and covered by i or j
                mask_i = (Cover[i_idx] == 1) & (cover_counts == 1)
                mask_j = (Cover[j_idx] == 1) & (cover_counts == 1)
                combined_critical = mask_i | mask_j
                combined_count = np.sum(combined_critical)
                
                if combined_count == 0:
                    # Both blocks are redundant
                    cover_counts -= Cover[i_idx]
                    cover_counts -= Cover[j_idx]
                    selected_indices.remove(i_idx)
                    selected_indices.remove(j_idx)
                    improved = True
                    break
                
                # Find a single block that covers all combined critical t-subsets
                # Use the first critical t-subset to get candidates
                c_idx = np.where(combined_critical)[0][0]
                candidates = inverted_index[c_idx]
                
                sel_set = set(selected_indices)
                found_replacement = False
                
                for cand_idx in candidates:
                    if cand_idx in sel_set:
                        continue
                    
                    # Check if cand covers all combined critical t-subsets
                    if np.sum(Cover[cand_idx] & combined_critical) == combined_count:
                        # Valid 2-for-1 exchange
                        cover_counts -= Cover[i_idx]
                        cover_counts -= Cover[j_idx]
                        cover_counts += Cover[cand_idx]
                        
                        selected_indices.remove(i_idx)
                        selected_indices.remove(j_idx)
                        selected_indices.append(cand_idx)
                        
                        improved = True
                        found_replacement = True
                        break
                
                if found_replacement:
                    selected_indices, cover_counts = remove_redundants(selected_indices, cover_counts)
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    # Final Redundancy Check
    selected_indices, cover_counts = remove_redundants(selected_indices, cover_counts)
    
    # Convert indices back to blocks
    blocks = [all_kblocks[i] for i in selected_indices]
    
    # Verification (Safety Check)
    # Verify that all t-subsets are covered
    if time.time() < deadline - 5.0:
        final_cover = np.zeros(n_tsubs, dtype=np.uint8)
        for idx in selected_indices:
            final_cover |= Cover[idx]
        if np.any(final_cover == 0):
            # Invalid - fall back to deterministic greedy
            selected_indices, _ = greedy_init(randomize=False)
            blocks = [all_kblocks[i] for i in selected_indices]
    
    return blocks
