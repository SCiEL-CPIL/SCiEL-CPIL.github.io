import itertools
import time
import numpy as np
import scipy.sparse as sp
import random
from math import comb
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # 1. Generate all t-subsets (requirements) and k-subsets (candidate blocks)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 2. Build sparse incidence matrix A (n_blocks x n_tsubs)
    # A[i, j] = 1 if tsubs[j] is a subset of blocks[i]
    row_indices = []
    col_indices = []
    data = []
    
    # Also precompute for each block, the list of t-subset indices it covers
    # This is crucial for fast updates in local search
    block_covered_tsubs = []
    
    # We need to map t-subsets to blocks for the 'remove' step logic potentially,
    # but actually for Add-Remove we just need to update counts.
    # Let's build block_covered_tsubs first.
    
    for i, block in enumerate(all_blocks):
        covered = []
        for ts in itertools.combinations(block, t):
            j = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            covered.append(j)
        block_covered_tsubs.append(covered)
        
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_tsubs), dtype=np.int8)
    
    # For fast access in greedy, we use the CSR structure
    # A.indptr, A.indices give the non-zero indices for each row
    
    def greedy_cover():
        """Fast greedy set cover using sparse matrix multiplication."""
        # u[j] = 1 if t-subset j is uncovered, 0 otherwise
        u = np.ones(n_tsubs, dtype=np.int8)
        selected_blocks = []
        
        # gains[i] = number of uncovered t-subsets covered by block i
        # Initially, gains[i] = sum(A[i, :]) = comb(k, t)
        # But we can just compute A @ u each step, which is fast enough for 120s?
        # n_blocks is C(19,8) = 75582. n_tsubs is C(19,5) = 11628.
        # A is sparse. A @ u is very fast.
        
        while True:
            if time.time() > deadline:
                break
            
            # Check if all covered
            # u.sum() == 0
            if not u.any():
                break
                
            # Calculate gains
            gains = A @ u
            
            # Find best block
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            # Pick the first block with max gain (deterministic for reproducibility, 
            # or we could randomize tie-breaking for diversity, but greedy is usually deterministic)
            # Using argmax gives the first one.
            best_idx = gains.argmax()
            
            selected_blocks.append(best_idx)
            
            # Update uncovered vector
            # Get indices of t-subsets covered by best_idx
            # In CSR, row i has non-zeros from indptr[i] to indptr[i+1]
            start = A.indptr[best_idx]
            end = A.indptr[best_idx + 1]
            cols = A.indices[start:end]
            u[cols] = 0
            
        return selected_blocks

    # Run initial greedy
    initial_solution = greedy_cover()
    
    if not initial_solution:
        return []
        
    # Convert to list of block indices
    selected_list = initial_solution[:]
    
    # Prepare state for local search
    # cover_count[j] = number of selected blocks covering t-subset j
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    in_solution = np.zeros(n_blocks, dtype=bool)
    
    for b_idx in selected_list:
        in_solution[b_idx] = True
        # Update cover counts
        start = A.indptr[b_idx]
        end = A.indptr[b_idx + 1]
        cols = A.indices[start:end]
        cover_count[cols] += 1
        
    best_len = len(selected_list)
    best_solution = selected_list[:]
    
    rng = random.Random(42)
    
    def try_remove_redundant():
        """
        Remove blocks from selected_list that are not the unique cover for any t-subset.
        Returns the number of blocks removed.
        """
        removed = 0
        i = 0
        while i < len(selected_list):
            b_idx = selected_list[i]
            # Check if any t-subset in this block has cover_count == 1
            # If yes, this block is essential.
            # If no, it is redundant.
            is_redundant = True
            start = A.indptr[b_idx]
            end = A.indptr[b_idx + 1]
            cols = A.indices[start:end]
            
            # Check if min(cover_count[cols]) > 1
            # If all are > 1, then removing this block won't uncover anything.
            if np.all(cover_count[cols] > 1):
                # Remove block
                in_solution[b_idx] = False
                cover_count[cols] -= 1
                selected_list.pop(i)
                removed += 1
                # Do not increment i, check the next element which is now at i
            else:
                i += 1
        return removed

    # Local Search Loop
    # Strategy: Add a random block, then remove redundancies.
    # If the total length decreases, update best.
    # If it increases too much, reset to best.
    
    time_limit = deadline - 5.0 # Leave 5s buffer for final cleanup/return
    
    while time.time() < time_limit:
        # Pick a random block to add
        # We should probably prioritize blocks that cover many currently uncovered or weakly covered sets?
        # But simple random is often effective in Add-Remove if the space is well-mixed.
        # Let's stick to random for simplicity, as the space of all blocks is fixed.
        b_add = rng.randint(0, n_blocks - 1)
        
        if in_solution[b_add]:
            continue
            
        # Add the block
        in_solution[b_add] = True
        selected_list.append(b_add)
        
        start = A.indptr[b_add]
        end = A.indptr[b_add + 1]
        cols = A.indices[start:end]
        cover_count[cols] += 1
        
        # Try to remove redundant blocks
        try_remove_redundant()
        
        current_len = len(selected_list)
        
        if current_len < best_len:
            best_len = current_len
            best_solution = selected_list[:]
            # Optional: restart the local search process from this new best?
            # Usually, we just continue from the current state which is now the best.
            
        # If we've drifted too far from the best, reset
        if current_len > best_len + 20:
            # Reset to best solution
            # Clear current state
            for b in selected_list:
                in_solution[b] = False
                start = A.indptr[b]
                end = A.indptr[b + 1]
                cols = A.indices[start:end]
                cover_count[cols] -= 1
            
            selected_list = best_solution[:]
            cover_count[:] = 0
            for b in best_solution:
                in_solution[b] = True
                start = A.indptr[b]
                end = A.indptr[b + 1]
                cols = A.indices[start:end]
                cover_count[cols] += 1

    # Convert best_solution indices to actual blocks
    return [list(all_blocks[i]) for i in best_solution]
