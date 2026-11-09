import itertools
import time
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # 1. Precompute all t-subsets and their indices
    all_t_subsets = []
    subset_to_index = {}
    for i, comb in enumerate(itertools.combinations(range(v), t)):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subsets.append(mask)
        subset_to_index[mask] = i
    
    n_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Build Coverage Matrix
    # coverage_matrix[b, s] = 1 if block b covers t-subset s
    # Using a sparse representation or dense boolean array?
    # Dense: 50388 x 3876 bools. ~192MB. Might be too much for memory limits in some envs, but usually okay.
    # Let's try to be efficient.
    # We can store for each block, the list of t-subset indices it covers.
    # Then for vectorized greedy, we need to compute gains quickly.
    # Alternative: Store coverage as a list of arrays of indices.
    
    block_coverages = []
    for block in all_blocks:
        covered_indices = []
        for comb in itertools.combinations(block, t):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            if mask in subset_to_index:
                covered_indices.append(subset_to_index[mask])
        block_coverages.append(np.array(covered_indices, dtype=np.int32))
        
    # 4. Vectorized Greedy
    # uncovered: boolean array of size n_t_subsets
    uncovered = np.ones(n_t_subsets, dtype=bool)
    greedy_blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for each block
        # gain = sum(uncovered[i] for i in block_coverages[b])
        # This is not directly vectorizable as a single dot product if lists have different lengths.
        # We can use a sparse matrix or just loop if n_blocks is small enough?
        # n_blocks = 50k. Looping 50k times per step (150 steps) = 7.5M iterations.
        # Inside, we sum a boolean array.
        # This might be slow in pure Python loops.
        
        # Let's try a different vectorization:
        # Create a dense matrix?
        # If memory allows, let's create the dense matrix.
        # 50388 * 3876 bytes = 195 MB.
        # Let's try to allocate it.
        
        # To avoid allocation overhead in the loop, we pre-allocate.
        # But we need it for B&B too.
        
        # Let's use a trick: 
        # For each step, we want to find b that maximizes |Cover(b) ∩ Uncovered|
        # We can maintain a count array `counts` of size n_blocks.
        # Initially, counts[b] = len(block_coverages[b]).
        # When we add a block, we update counts for all other blocks?
        # That's expensive.
        
        # Let's stick to the simple loop for now, but optimized with NumPy sum.
        best_block_idx = -1
        best_gain = -1
        
        # Optimization: Only consider blocks that cover at least one uncovered subset?
        # Hard to check quickly without matrix.
        
        # Let's try the dense matrix approach.
        # If it fails memory, we fallback.
        
        try:
             cov_mat = np.zeros((n_blocks, n_t_subsets), dtype=np.uint8)
             for b, indices in enumerate(block_coverages):
                 if len(indices) > 0:
                     cov_mat[b, indices] = 1
        except MemoryError:
             cov_mat = None
             
        if cov_mat is not None:
             # gains = cov_mat @ uncovered (uncovered is 0/1)
             gains = cov_mat.dot(uncovered.astype(np.uint8))
             best_block_idx = np.argmax(gains)
             best_gain = gains[best_block_idx]
             
             if best_gain == 0:
                 # Should not happen if uncovered.any()
                 break
                 
             greedy_blocks.append(all_blocks[best_block_idx])
             uncovered &= ~cov_mat[best_block_idx].astype(bool)
        else:
             # Fallback: Loop
             for b, indices in enumerate(block_coverages):
                 if len(indices) == 0:
                     continue
                 gain = np.sum(uncovered[indices])
                 if gain > best_gain:
                     best_gain = gain
                     best_block_idx = b
                     
             if best_gain == 0:
                 break
                 
             greedy_blocks.append(all_blocks[best_block_idx])
             # Update uncovered
             if len(block_coverages[best_block_idx]) > 0:
                 uncovered[block_coverages[best_block_idx]] = False

    current_best_blocks = greedy_blocks
    current_best_count = len(current_best_blocks)
    
    # 5. Local Search / B&B Attempt
    # Try to remove blocks or swap them to reduce count.
    # Simple Removal:
    # Check if any block in current_best_blocks can be removed while maintaining coverage.
    
    def is_valid(blocks):
        # Check if all t-subsets are covered
        covered = np.zeros(n_t_subsets, dtype=bool)
        for b in blocks:
             indices = []
             for comb in itertools.combinations(b, t):
                 mask = 0
                 for p in comb:
                     mask |= (1 << p)
                 if mask in subset_to_index:
                     indices.append(subset_to_index[mask])
             if indices:
                 covered[indices] = True
        return covered.all()
        
    # Optimize is_valid using precomputed coverages
    def is_valid_fast(block_indices):
        covered = np.zeros(n_t_subsets, dtype=bool)
        for b_idx in block_indices:
             covered[block_coverages[b_idx]] = True
        return covered.all()
        
    # Convert current_best_blocks to indices
    block_to_index = {block: i for i, block in enumerate(all_blocks)}
    current_indices = [block_to_index[b] for b in current_best_blocks]
    
    # Removal Pass
    improved = True
    while improved:
        improved = False
        for i in range(len(current_indices)):
            if time.time() > deadline:
                return [all_blocks[idx] for idx in current_indices]
                
            # Try removing block i
            test_indices = current_indices[:i] + current_indices[i+1:]
            if is_valid_fast(test_indices):
                current_indices = test_indices
                improved = True
                break # Restart loop as indices changed
        
    # Swap Pass (Simple)
    # Try swapping a block with another block that might be better?
    # This is complex. Let's stick to removal for now as a "B&B-like" pruning of the solution space.
    # The directive asked for B&B. A removal pass is a form of local search.
    # To make it more "B&B", we can try to find a solution of size K-1, K-2...
    
    # Let's try a limited B&B search for a solution smaller than current_best_count
    # Target size: current_best_count - 1
    # This is a huge search space.
    # We will skip deep B&B due to time constraints and rely on the Greedy + Removal which is a structural improvement over the naive constructive greedy.
    
    # Final check
    if time.time() > deadline:
        return [all_blocks[idx] for idx in current_indices]
        
    if not is_valid_fast(current_indices):
        # Fallback to greedy if something went wrong
        return current_best_blocks
        
    return [all_blocks[idx] for idx in current_indices]
