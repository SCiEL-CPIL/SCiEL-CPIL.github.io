import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    # Map t-subsets to indices for fast lookup
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute the t-subsets contained in each k-subset
    # t_count is the number of t-subsets in a k-set
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        return int(np.prod([n-i for i in range(r)]) / np.prod([i+1 for i in range(r)]))
        
    t_count = nCr(k, t)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    # This allows fast gain calculation
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Greedy Construction with Randomization
    covered = np.zeros(n_ts, dtype=bool)
    blocks_indices = [] # Store indices into all_ks
    point_load = np.zeros(v, dtype=int)
    
    # We will run a greedy loop
    # Optimization: Instead of checking all n_ks every time (which is expensive for v=24, k=7),
    # we can just do it. n_ks = C(24,7) = 346104.
    # t_count = C(7,3) = 35.
    # Memory for block_t_indices: 346104 * 35 * 4 bytes ~ 48 MB. This is fine.
    
    # The main cost is the greedy step: calculating gains for all 346k blocks.
    # gains = (~covered[block_t_indices]).sum(axis=1)
    # This involves indexing a boolean array of size n_ts with a matrix of size n_ks*t_count.
    # Resulting temp array size: 346104 * 35 ~ 12 million booleans.
    # This is fast in NumPy.
    
    while time.time() - start_time < time_limit * 0.6:
        # Calculate gains
        # covered[block_t_indices] creates a boolean matrix
        # ~covered[...] inverts it
        # .sum(axis=1) sums the True values (uncovered subsets)
        
        # To save time, we can break early if all are covered?
        # No, we want to minimize blocks, so we stop when we can't cover more?
        # Actually, greedy stops when max gain is 0.
        
        uncovered_flags = ~covered[block_t_indices]
        gains = uncovered_flags.sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find candidates with max gain
        # To add randomness and escape local optima, pick from top K candidates randomly
        # Or just pick the one with min point load sum among max gain candidates?
        # Let's stick to a simple random selection from the top 10 max gain candidates to diversify
        # if time permits, otherwise just max.
        
        candidates = np.where(gains == max_gain)[0]
        
        # Tie-breaking: Prefer blocks with lower load to balance coverage
        # Calculate load sum for candidates
        # all_ks is a list of tuples. We need to check loads.
        # This might be slow if done in Python loop.
        # Let's just pick a random candidate from the max gain set.
        # Randomization helps avoid deterministic traps.
        best_idx = candidates[np.random.randint(len(candidates))]
        
        # Add block
        blocks_indices.append(best_idx)
        block = all_ks[best_idx]
        
        # Update covered
        covered[block_t_indices[best_idx]] = True
        
        # Update point load
        for p in block:
            point_load[p] += 1
            
        # Check if all covered
        if np.all(covered):
            break
            
    # Local Search: Try to remove redundant blocks
    # This can significantly reduce the count.
    # A block is redundant if all its t-subsets are covered by other blocks.
    
    # We need to count how many blocks cover each t-subset.
    # coverage_count[j] = number of blocks in blocks_indices that contain t-subset j
    
    # Reset coverage count
    coverage_count = np.zeros(n_ts, dtype=np.int32)
    
    # Build initial coverage count
    for b_idx in blocks_indices:
        coverage_count[block_t_indices[b_idx]] += 1
        
    # Iterate through blocks and try to remove
    # We need to be careful with order. 
    # Simple approach: iterate through indices, if a block is redundant (min coverage in its subsets is > 1), remove it.
    # However, removing one block changes the redundancy of others.
    # A standard heuristic is to scan repeatedly until no more removals are possible.
    
    current_blocks = list(blocks_indices)
    
    # To speed up, we can just do a few passes.
    # Time budget for local search: remaining time
    ls_start = time.time()
    ls_time_limit = time_limit - (ls_start - start_time) - 5.0 # 5s safety
    
    improved = True
    while improved and (time.time() - ls_start) < ls_time_limit:
        improved = False
        new_current_blocks = []
        # Rebuild coverage count for current_blocks? 
        # It's better to maintain it incrementally or rebuild if list is small.
        # Rebuilding is O(N_blocks * t_count). N_blocks ~ 76. t_count ~ 35. Very fast.
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        for b_idx in current_blocks:
            # Check if b_idx is redundant
            # If all t-subsets in b_idx have coverage_count > 1, it can be removed.
            # Note: coverage_count includes b_idx itself.
            # So we need min(coverage_count[block_t_indices[b_idx]]) > 1
            subs_in_block = block_t_indices[b_idx]
            min_cov = np.min(coverage_count[subs_in_block])
            
            if min_cov > 1:
                # Redundant, remove it
                improved = True
                # Decrement coverage for this block
                coverage_count[subs_in_block] -= 1
            else:
                # Keep it
                new_current_blocks.append(b_idx)
                
        current_blocks = new_current_blocks
        
    # Convert indices back to blocks
    final_blocks = [list(all_ks[i]) for i in current_blocks]
    
    return final_blocks
