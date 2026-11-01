import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using a randomized greedy set cover approach.
    """
    # For this specific task v=24, k=7, t=3.
    # We can hardcode optimizations for this specific case if needed, 
    # but the generic algorithm should work well.
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for faster lookup
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # We need to evaluate candidate blocks.
    # Generating all C(24, 7) = 346,104 blocks upfront might be memory heavy if we store all their coverage sets,
    # but it's feasible. However, calculating gain dynamically is better.
    
    # A better strategy: 
    # 1. Maintain a set of uncovered t-subsets.
    # 2. At each step, find the block that covers the most uncovered t-subsets.
    #    To do this efficiently, we can iterate through potential blocks? No, too slow.
    #    Instead, we can use a heuristic: 
    #    Pick a random uncovered t-subset. Generate random k-sets containing it?
    #    Or, use a "Greedy with Randomization" where we pick the best block from a pool of candidates.
    
    # Let's use a standard Greedy Set Cover with random restarts.
    # To speed up "find best block", we can precompute the list of all k-subsets?
    # C(24,7) is ~346k. For each block, precomputing its t-subsets is 35 per block.
    # Total triples stored: 346k * 35 ~ 12 million ints. This is fine.
    
    # However, checking gain for every block at every step is O(NumBlocks * t) per step.
    # NumBlocks ~ 346k. Steps ~ 76. Total ops ~ 26M * 3 = 78M. This is fast enough in Python/NumPy if vectorized?
    # Maybe not pure Python. Let's try to optimize.
    
    # Alternative: 
    # 1. Generate a pool of candidate blocks.
    # 2. Use a priority queue or simply scan the pool to find the max gain.
    
    # Let's try a robust approach:
    # 1. Precompute all k-subsets and their constituent t-subsets (as indices).
    # 2. Use a loop with random restarts.
    # 3. In each run, maintain an array `uncovered` of booleans.
    # 4. At each step, find the block with max count of True in its t-subset list.
    
    # Precomputation
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # For each k-subset, store the indices of the t-subsets it covers
    # This is a list of lists of ints.
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    # Convert to numpy array for faster summing? 
    # block_cover_indices is a list of lists of varying length? No, all length C(k,t).
    # C(7,3) = 35. So it's a fixed size.
    # Let's make it a 2D numpy array: shape (n_k_subsets, C(k,t))
    c_k_t = len(block_cover_indices[0])
    cover_matrix = np.array(block_cover_indices, dtype=np.int32)
    
    best_blocks = []
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    # Number of restarts depends on speed. Let's try 50-100 restarts.
    # Each restart takes ~76 steps.
    # In each step, we need to find the max gain.
    # Gain for block i = sum(uncovered[cover_matrix[i]])
    # We can vectorize this? 
    # uncovered is a boolean array of size n_t_subsets.
    # We want for each row in cover_matrix, the sum of uncovered[row].
    # This is effectively a matrix multiplication or advanced indexing.
    # np.sum(uncovered[cover_matrix], axis=1) -> This creates a large intermediate array (n_k_subsets * 35).
    # 346k * 35 ~ 12M elements. This is fast in NumPy.
    
    for restart in range(100):
        if time.time() - start_time > timeout:
            break
            
        random.seed(restart)
        
        # Initialize uncovered
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found = []
        
        # Optional: Randomly shuffle the order in which we consider ties or break ties?
        # The greedy choice is deterministic given the uncovered set. 
        # To diversify, we can add a small random noise to the gain or randomly select from top-K.
        # Let's stick to pure greedy first, restarts handle the randomness.
        
        while True:
            if not np.any(uncovered):
                break
                
            # Calculate gain for all blocks
            # uncovered[cover_matrix] gets the values, sum along axis 1
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            # Find the block with max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                # Should not happen if uncovered is not empty and we have all blocks
                break
                
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly to ensure diversity across restarts even if gains are identical
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            
            block = all_k_subsets[chosen_idx]
            blocks_found.append(block)
            
            # Update uncovered
            uncovered[cover_matrix[chosen_idx]] = False
            
        if len(blocks_found) < best_count:
            best_count = len(blocks_found)
            best_blocks = blocks_found
            # If we hit the theoretical lower bound or known best, we can stop?
            # Known best is 76. If we get 76, we might want to continue to see if we can get lower?
            # But 76 is the current best known. Let's assume 76 is a good target.
            # If we find 76, we can stop early to save time? No, maybe 75 exists? Unlikely but possible.
            # Let's just keep searching until time runs out.
            
    return best_blocks
