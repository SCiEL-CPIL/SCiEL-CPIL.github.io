import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets and k-blocks
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subset to index
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute coverage: for each block, list of t-subset indices it covers
    # This is the core data structure
    block_covered_indices = []
    for b in all_blocks:
        idxs = []
        for sub in itertools.combinations(b, t):
            idxs.append(subset_to_idx[sub])
        block_covered_indices.append(idxs)
    
    # Convert to numpy arrays for fast operations
    # We will maintain a boolean array `covered` of size n_subsets
    # For each block, we have a boolean mask of size n_subsets indicating what it covers
    # But n_blocks * n_subsets is 77520 * 3876 ~ 300M booleans ~ 300MB. This is feasible.
    # Let's create a matrix M where M[i, j] is 1 if block i covers subset j.
    # Actually, iterating through 77k blocks and setting 35 elements each is fast enough in Python if we do it once.
    
    # Let's use a list of numpy arrays or just lists of indices?
    # Using a sparse representation (list of indices) is memory efficient.
    # To speed up the greedy step:
    # We need to find argmax of (number of newly covered subsets).
    # new_covered_i = sum(covered[j] is False for j in block_covered_indices[i])
    
    # Strategy:
    # 1. Initialize `covered` as np.zeros(n_subsets, dtype=bool)
    # 2. In each greedy step, for each block i, compute gain = len(block_covered_indices[i]) - sum(covered[j] for j in block_covered_indices[i])
    # 3. Pick block with max gain.
    
    # Optimization: 
    # Pre-compute `block_covered_indices` as numpy arrays of int64.
    # `covered` is a boolean numpy array.
    # For each block i: gain_i = np.sum(~covered[block_covered_indices[i]])
    # This is still O(n_blocks * t_subsets_per_block) per step.
    # With ~150 blocks in final solution, and ~35 subsets per block, this is 77520 * 35 ~ 2.7M operations per step.
    # 150 steps * 2.7M = 400M operations. In Python, this might be too slow (1-2 seconds per step?).
    # We need to be faster.
    
    # Alternative: Use a bitset approach with numpy integers?
    # Or, pre-compute a matrix A of shape (n_blocks, n_subsets) with 0s and 1s?
    # Memory: 77520 * 3876 bytes ~ 300MB. This is okay.
    # Then gain_vector = A.dot(~covered)  # This is a matrix-vector product.
    # A is (n_blocks, n_subsets), ~covered is (n_subsets,). Result is (n_blocks,).
    # NumPy dot product is highly optimized (BLAS).
    # Let's try this.
    
    # Build A
    A = np.zeros((n_blocks, n_subsets), dtype=np.int8)
    for i, idxs in enumerate(block_covered_indices):
        A[i, idxs] = 1
        
    # Function to perform one greedy run
    def greedy_run(seed):
        rng = random.Random(seed)
        covered = np.zeros(n_subsets, dtype=bool)
        selected_blocks = []
        
        # Initial randomization: pick a few random blocks to break symmetry?
        # Or just start greedy.
        
        while not covered.all():
            # Calculate gains
            # gain_i = sum(1 for j in A[i] if not covered[j])
            # gain = A.dot(~covered)
            gains = A.dot(~covered)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break # No block can cover any new subset
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly among candidates
            chosen_idx = rng.choice(candidates)
            
            selected_blocks.append(chosen_idx)
            covered[A[chosen_idx] == 1] = True
            
            # Time check
            if time.time() > deadline - 5:
                return None # Timeout, return None to indicate failure
        
        return selected_blocks

    # We will run multiple greedy restarts with different seeds.
    # Each run takes roughly T seconds. We can fit ~ (deadline - start_time - setup_time) / T runs.
    # Let's estimate T. 
    # Setup A took some time. 
    # One greedy run: ~150 steps. Each step: dot product of (77520, 3876) and (3876,).
    # This is ~ 300M multiply-adds. Modern CPU does ~ 10-50 GFLOPS. So ~ 10-30 ms per step?
    # 150 steps * 20ms = 3 seconds per run.
    # We have ~130 seconds. So we can do ~ 40 runs.
    
    best_solution = None
    best_size = float('inf')
    
    # Run 1: Standard greedy
    sol = greedy_run(0)
    if sol:
        best_solution = sol
        best_size = len(sol)
    
    # Run more with different seeds to explore different local minima
    seed = 1
    while time.time() < deadline - 10:
        sol = greedy_run(seed)
        if sol:
            if len(sol) < best_size:
                best_size = len(sol)
                best_solution = sol
        seed += 1
        
    # If no solution found (timeout or logic error), fallback to a simple valid (though possibly large) construction
    # But greedy should always find one if the covering exists.
    # If best_solution is None, we have a problem.
    if best_solution is None:
        # Emergency fallback: just return a large random set? No, must be valid.
        # Let's force one more run with a longer deadline if we have time, or just accept failure.
        # Actually, if greedy fails, it means we hit the time limit mid-run.
        # We should probably return the best partial result? No, must be valid covering.
        # The contract says "Valid when every t-subset... lies in at least one block".
        # If we can't find a valid one, we return [] which scores 0.
        # But greedy is very likely to succeed.
        
        # Let's try to finish the last run if it was close?
        # No, let's just return [] if no valid solution found.
        return []
        
    # Convert indices to actual blocks
    result_blocks = [list(all_blocks[idx]) for idx in best_solution]
    return result_blocks

