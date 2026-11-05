import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Deadline with 10s safety margin
    deadline = start_time + 110.0
    
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    points = range(v)
    all_t_subs = list(itertools.combinations(points, t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Generate all k-subsets (blocks)
    all_k_subs = list(itertools.combinations(points, k))
    n_k = len(all_k_subs)
    
    # Precompute coverage relationships
    # blocks_to_subsets[b] = list of indices of t-subsets covered by block b
    # subsets_to_blocks[s] = list of indices of k-blocks covering subset s
    blocks_to_subsets = [[] for _ in range(n_k)]
    subsets_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, b in enumerate(all_k_subs):
        # A k-subset covers all its t-subsets
        for s in itertools.combinations(b, t):
            s_idx = t_sub_to_idx[s]
            blocks_to_subsets[b_idx].append(s_idx)
            subsets_to_blocks[s_idx].append(b_idx)
            
    # To speed up random tie-breaking, we might not need to store all candidates,
    # just track max and count? No, random choice needs the list.
    # But n_k is small enough.

    best_result = None
    best_size = float('inf')
    
    # Run multiple restarts to find a better covering
    # The number of restarts is limited by time
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        
        # Greedy Construction
        # uncovered_count[b] = number of *currently uncovered* t-subsets that block b covers
        uncovered_count = [len(blocks_to_subsets[i]) for i in range(n_k)]
        covered = [False] * n_t
        solution_indices = []
        
        # Greedy loop
        while True:
            if time.time() > deadline - 2.0:
                break
                
            max_val = -1
            best_b = -1
            # To allow randomization in ties, we can just pick the first max or random max.
            # Scanning all n_k is O(n_k).
            # Optimization: If we used a heap, we'd have to update it.
            # Scanning is simple and fast for n_k ~ 75k.
            
            # Find max uncovered_count
            for b in range(n_k):
                if uncovered_count[b] > max_val:
                    max_val = uncovered_count[b]
                    best_b = b
                    # We don't break here to allow for potential random tie-breaking?
                    # Actually, finding *all* maxes is slower.
                    # Let's just pick the first one found with max value.
                    # To introduce randomness, we can randomize the order of b?
                    # Or, simpler: just run deterministic greedy, and rely on reduction.
                    # But random restarts help.
                    # Let's try: if c == max_val, we can randomly swap?
                    # No, let's stick to deterministic for speed, but add randomization in the 
                    # *order* of processing blocks? No, blocks are static.
                    
            if max_val == 0:
                break
                
            solution_indices.append(best_b)
            
            # Update coverage
            for s_idx in blocks_to_subsets[best_b]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    # Decrement count for all other blocks covering this subset
                    for b_prime in subsets_to_blocks[s_idx]:
                        uncovered_count[b_prime] -= 1
                        
        # If we timed out during greedy, skip reduction for this iteration
        if not any(not c for c in covered):
            continue
            
        # Reduction Phase
        # Initialize cover counts for the greedy solution
        final_cover_count = [0] * n_t
        for b in solution_indices:
            for s_idx in blocks_to_subsets[b]:
                final_cover_count[s_idx] += 1
                
        sol_set = set(solution_indices)
        
        # Remove redundant blocks
        # We iterate until no more blocks can be removed
        while True:
            if time.time() > deadline - 1.0:
                break
                
            changed = False
            # Iterate over a list copy of the set
            for b in list(sol_set):
                # Check if any t-subset in b is covered ONLY by b
                essential = False
                for s_idx in blocks_to_subsets[b]:
                    if final_cover_count[s_idx] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove b
                    sol_set.remove(b)
                    for s_idx in blocks_to_subsets[b]:
                        final_cover_count[s_idx] -= 1
                    changed = True
                    
            if not changed:
                break
                
        # Convert to point lists
        current_result = [list(all_k_subs[b]) for b in sol_set]
        
        if len(current_result) < best_size:
            best_size = len(current_result)
            best_result = current_result

    if best_result is None:
        # Fallback: just return the first greedy solution or empty?
        # Should not happen.
        return []
        
    return best_result

# EVOLVE-BLOCK-END
