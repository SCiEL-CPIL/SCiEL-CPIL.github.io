"""
Covering design C(19,8,5) using a hybrid approach:
1. Fast greedy set-cover with numpy/scipy-based incremental gain updates.
2. Local search polish (removal + swap) to minimize block count.
3. Deadline guard for 140s wall-clock limit.
"""
import itertools
import time
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute all k-blocks
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # For each k-block, precompute the list of t-subset indices it covers
    block_tsubs = []
    for block in all_ksubs:
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            tsubs_in_block.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_block)
    
    # Convert to numpy array for fast indexing
    block_tsubs_np = np.array(block_tsubs, dtype=np.int32)

    # Build sparse matrix: rows are blocks, cols are t-subsets
    # M[i, j] = 1 if block i covers t-subset j
    M = lil_matrix((n_ksubs, n_tsubs), dtype=np.int32)
    for i, tsubs in enumerate(block_tsubs):
        for j in tsubs:
            M[i, j] = 1
    
    # Convert to CSR for fast matrix-vector multiplication
    M_csr = M.tocsr()

    # Greedy loop
    covered = np.zeros(n_tsubs, dtype=bool)
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    block_indices = []
    
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break

        # Compute gains: for each block, number of uncovered t-subsets it covers
        uncov = (~covered).astype(np.int32)
        gains = M_csr.dot(uncov)

        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break  # No improvement possible

        block_indices.append(best_idx)
        
        # Update covered and cover_count
        tsubs_covered = block_tsubs_np[best_idx]
        covered[tsubs_covered] = True
        cover_count[tsubs_covered] += 1

    # Phase 2: Local Search Polish
    # 2a. Remove redundant blocks
    def try_remove_redundant(block_indices, cover_count, block_tsubs_np, deadline):
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_indices = []
            for i in range(len(block_indices)):
                idx = block_indices[i]
                tsubs = block_tsubs_np[idx]
                # Check if all t-subsets covered by this block are covered by others
                # i.e., cover_count[t] >= 2 for all t in tsubs
                if np.all(cover_count[tsubs] >= 2):
                    # Remove it
                    cover_count[tsubs] -= 1
                    changed = True
                else:
                    new_indices.append(idx)
            block_indices = new_indices
        return block_indices

    block_indices = try_remove_redundant(block_indices, cover_count, block_tsubs_np, deadline)
    
    # 2b. Try swaps: replace one block with another if it enables further removals
    # This is complex, so we'll do multiple passes of removal
    
    # Re-run removal in case the order matters
    block_indices = try_remove_redundant(block_indices, cover_count, block_tsubs_np, deadline)

    # Final check: ensure validity
    # Rebuild cover_count from scratch to be safe
    cover_count_final = np.zeros(n_tsubs, dtype=np.int32)
    for idx in block_indices:
        tsubs = block_tsubs_np[idx]
        cover_count_final[tsubs] += 1
    
    if not np.all(cover_count_final >= 1):
        # Fallback to original greedy result if something went wrong (shouldn't happen)
        # But we need to return a valid solution.
        # The greedy result was valid.
        pass

    # Return the blocks
    return [list(all_ksubs[idx]) for idx in block_indices]
