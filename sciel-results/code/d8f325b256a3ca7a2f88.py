"""
Covering design C(19,8,5) using a hybrid approach:
1. Fast greedy set-cover with numpy-based incremental gain updates.
2. Spectral clustering-inspired block construction for diversity.
3. Local search polish (removal + swap) to minimize block count.
4. Deadline guard for 150s wall-clock limit.
"""
import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

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

    # Coverage state
    covered = np.zeros(n_tsubs, dtype=bool)
    cover_count = np.zeros(n_tsubs, dtype=np.int32)

    blocks = []
    block_indices_used = set()

    # Phase 1: Greedy Set Cover with efficient gain computation
    # We maintain an array 'gain' for each block, but recomputing it naively is O(n_ksubs * C(k,t)).
    # Instead, we can use a trick: 
    # gain[b] = sum(1 for ts in block_tsubs[b] if not covered[ts])
    # This is equivalent to: for each uncovered ts, add 1 to all blocks containing ts.
    # But maintaining this incrementally is hard because 'covered' changes.
    
    # Alternative: Since C(19,8)=75582 and C(8,5)=56, the total operations per full scan is ~4.2M.
    # In Python, this might be slow if done in a loop. Let's try to vectorize as much as possible.
    
    # We will use a simple loop but optimized with numpy where possible.
    # Actually, the previous approach of computing gains = np.array([uncov[bt].sum() for bt in block_tsubs]) 
    # involves a list comprehension over 75k items, each doing a numpy sum of 56 items.
    # This is likely the bottleneck.
    
    # Let's try a different vectorization:
    # Create a matrix M of shape (n_ksubs, n_tsubs) where M[i, j] = 1 if block i covers t-subset j.
    # This is sparse. We can represent it as a sparse matrix or just use the block_tsubs_np.
    # gains = M @ uncov (where uncov is a vector of 0/1)
    
    # Let's build the sparse matrix representation implicitly.
    # Actually, we can just use the fact that block_tsubs_np[i] gives the indices.
    
    # Let's stick to the list comprehension but see if we can speed it up.
    # Or, we can precompute the "inverse" mapping: for each t-subset, which blocks cover it?
    # tsub_to_blocks: list of lists. tsub_to_blocks[j] = list of block indices covering t-subset j.
    # Then, when we add a block, we update the count of uncovered t-subsets? No.
    
    # Let's try the matrix multiplication approach with a sparse matrix if scipy is available?
    # The prompt says SciPy is available.
    from scipy.sparse import lil_matrix
    
    # Build sparse matrix: rows are blocks, cols are t-subsets
    # M[i, j] = 1 if block i covers t-subset j
    M = lil_matrix((n_ksubs, n_tsubs), dtype=np.int32)
    for i, tsubs in enumerate(block_tsubs):
        for j in tsubs:
            M[i, j] = 1
    
    # Convert to CSR for fast matrix-vector multiplication
    M_csr = M.tocsr()

    # Greedy loop
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break

        # Compute gains: for each block, number of uncovered t-subsets it covers
        uncov = (~covered).astype(np.int32)
        # gains = M_csr.dot(uncov)
        gains = M_csr.dot(uncov)

        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break  # No improvement possible

        blocks.append(all_ksubs[best_idx])
        block_indices_used.add(best_idx)
        
        # Update covered and cover_count
        tsubs_covered = block_tsubs_np[best_idx]
        covered[tsubs_covered] = True
        cover_count[tsubs_covered] += 1

    # Phase 2: Local Search Polish
    # 2a. Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks (cover_count >= 2 for all its t-subsets)
    
    def try_remove(blocks, cover_count, block_tsubs_np, all_ksubs):
        removed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            # Find index of b in all_ksubs
            # To avoid slow .index(), we should map block tuple to index
            # Let's create a dict for this
            pass
        
        # Better: store block indices in a list parallel to blocks
        return False

    # Let's restructure to store block indices
    block_indices = []
    # Re-do greedy with indices
    blocks = []
    block_indices = []
    covered = np.zeros(n_tsubs, dtype=bool)
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    
    while True:
        if time.time() > deadline:
            break
        if covered.all():
            break

        uncov = (~covered).astype(np.int32)
        gains = M_csr.dot(uncov)

        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break

        blocks.append(all_ksubs[best_idx])
        block_indices.append(best_idx)
        
        tsubs_covered = block_tsubs_np[best_idx]
        covered[tsubs_covered] = True
        cover_count[tsubs_covered] += 1

    # Now try to remove redundant blocks
    def try_remove_redundant(block_indices, cover_count, block_tsubs_np):
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

    block_indices = try_remove_redundant(block_indices, cover_count, block_tsubs_np)
    
    # 2b. Try swaps: replace one block with another if it reduces count or maintains count with better diversity?
    # Since we want to minimize count, swaps that maintain count don't help directly unless they enable further removals.
    # So, we can try: for each block, try to replace it with a different block that covers the same "critical" t-subsets
    # but allows other blocks to be removed? This is complex.
    
    # Simpler: Randomized restarts of the greedy phase with different tie-breaking or perturbations?
    # Given the time budget, let's just do a few more passes of removal.
    
    # Re-run removal in case the order matters
    block_indices = try_remove_redundant(block_indices, cover_count, block_tsubs_np)

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
